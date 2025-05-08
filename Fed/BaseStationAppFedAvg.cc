#include "BaseStationAppFedAvg.h"
#include "inet/common/ModuleAccess.h"
#include "inet/common/TimeTag_m.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/transportlayer/contract/udp/UdpControlInfo_m.h"
#include "inet/networklayer/common/L3AddressTag_m.h"
#include "inet/common/packet/Packet.h"
#include "inet/common/packet/chunk/Chunk.h"
#include "inet/common/Ptr.h"

Define_Module(BaseStationAppFedAvg);

simsignal_t BaseStationAppFedAvg::rcvdPkSignal = registerSignal("rcvdPk");

BaseStationAppFedAvg::BaseStationAppFedAvg() : globalModel(5, 0.01, 32, 3) {
}

BaseStationAppFedAvg::~BaseStationAppFedAvg() {
    cancelAndDelete(roundTimer);
}

void BaseStationAppFedAvg::initialize(int stage) {
    ApplicationBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        localPort = par("localPort");
        numUavs = par("numUavs");
        fedAvgPort = par("fedAvgPort");
        maxRounds = par("maxRounds");
        roundInterval = par("roundInterval");

        numReceived = 0;
        currentRound = 0;

        roundCompletedSignal = registerSignal("roundCompleted");
        modelAccuracySignal = registerSignal("modelAccuracy");

        WATCH(numReceived);
        WATCH(currentRound);
    }
    else if (stage == INITSTAGE_APPLICATION_LAYER) {
        socket.setOutputGate(gate("socketOut"));
        socket.setCallback(this);
        
        // Only bind the socket here, not in handleStartOperation as well
        socket.bind(localPort);

        // Initialiser le timer pour la première ronde
        roundTimer = new cMessage("roundTimer");
        scheduleAt(simTime() + par("startTime"), roundTimer);

        // Initialiser le modèle global
        globalModel.initializeWeights();

        EV_INFO << "Base Station FedAvg initialized. Ready to start federated learning with "
                << numUavs << " UAVs." << endl;
    }
}

void BaseStationAppFedAvg::handleMessageWhenUp(cMessage *msg) {
    if (msg->isSelfMessage()) {
        if (msg == roundTimer) {
            startNextRound();
        }
    }
    else {
        socket.processMessage(msg);
    }
}

void BaseStationAppFedAvg::startNextRound() {
    currentRound++;

    if (currentRound <= maxRounds) {
        EV_INFO << "Starting federated learning round " << currentRound << "/" << maxRounds << endl;

        // Réinitialiser les modèles reçus pour cette ronde
        receivedModels.clear();
        samplesPerUav.clear();

        // Diffuser le modèle global à tous les UAVs
        broadcastGlobalModel();

        // Planifier la prochaine ronde
        scheduleAt(simTime() + roundInterval, roundTimer);
    }
    else {
        EV_INFO << "Federated learning completed after " << (currentRound-1) << " rounds." << endl;
    }
}

void BaseStationAppFedAvg::broadcastGlobalModel() {
    // Créer un paquet avec le message FedAvg
    char msgName[32];
    sprintf(msgName, "GlobalModel-Round%d", currentRound);
    Packet *packet = new Packet(msgName);

    // Créer le message FedAvg
    const auto& fedAvgMsg = makeShared<FedAvgMessage>();
    fedAvgMsg->setMessageType(GLOBAL_UPDATE);
    fedAvgMsg->setRoundId(currentRound);
    fedAvgMsg->setModelWeights(globalModel.serialize().c_str());
    fedAvgMsg->setUavId(-1);  // -1 signifie station de base

    // Ajouter le message au paquet
    packet->insertAtBack(fedAvgMsg);

    // Diffuser le paquet à tous les UAVs (via multicast ou multiples unicast)
    for (int i = 0; i < numUavs; i++) {
        // Dans un cas réel, nous utiliserions des adresses multicast ou nous enverrions
        // à chaque UAV individuellement en utilisant leur adresse
        std::string destAddr = "uav[" + std::to_string(i) + "]";
        L3Address destAddress;
        L3AddressResolver().tryResolve(destAddr.c_str(), destAddress);

        if (!destAddress.isUnspecified()) {
            // Créer une copie du paquet pour chaque UAV
            Packet *pktCopy = packet->dup();
            socket.sendTo(pktCopy, destAddress, fedAvgPort);
            EV_INFO << "Sent global model to UAV " << i << " for round " << currentRound << endl;
        }
    }

    // Supprimer le paquet original
    delete packet;
}

void BaseStationAppFedAvg::aggregateModels() {
    if (receivedModels.empty()) {
        EV_WARN << "No models received for aggregation in round " << currentRound << endl;
        return;
    }

    EV_INFO << "Aggregating " << receivedModels.size() << " models for round " << currentRound << endl;

    // Calculer le nombre total d'échantillons
    int totalSamples = 0;
    for (const auto& entry : samplesPerUav) {
        totalSamples += entry.second;
    }

    if (totalSamples == 0) {
        EV_WARN << "Total samples count is zero, cannot perform weighted aggregation" << endl;
        return;
    }

    // Obtenir les dimensions du modèle à partir du premier modèle
    FederatedLearningModel tempModel;
    tempModel.deserialize(receivedModels.begin()->second);
    std::vector<double> aggregatedWeights(tempModel.getWeights().size(), 0.0);

    // Agréger les modèles avec une moyenne pondérée
    for (const auto& entry : receivedModels) {
        int uavId = entry.first;
        const std::string& modelStr = entry.second;
        int samples = samplesPerUav[uavId];

        // Désérialiser le modèle
        FederatedLearningModel uavModel;
        if (uavModel.deserialize(modelStr)) {
            // Pondérer par le nombre d'échantillons
            double weight = static_cast<double>(samples) / totalSamples;
            const auto& uavWeights = uavModel.getWeights();

            // Ajouter à la moyenne pondérée
            for (size_t i = 0; i < aggregatedWeights.size(); i++) {
                aggregatedWeights[i] += uavWeights[i] * weight;
            }
        }
    }

    // Mettre à jour le modèle global avec les poids agrégés
    globalModel.setWeights(aggregatedWeights);

    // Émettre un signal de progression
    emit(roundCompletedSignal, currentRound);

    EV_INFO << "Model aggregation completed for round " << currentRound << endl;
}

void BaseStationAppFedAvg::socketDataArrived(UdpSocket *socket, Packet *packet) {
    // Traitement des données reçues des UAVs
    auto addressInd = packet->getTag<L3AddressInd>();
    L3Address srcAddr = addressInd->getSrcAddress();

    // Calculer le délai de bout en bout
    auto creationTimeTag = packet->getTag<CreationTimeTag>();
    simtime_t delay = simTime() - creationTimeTag->getCreationTime();

    EV_INFO << "Received packet " << packet->getName() << " from UAV at "
            << srcAddr.str() << ". Delay: " << delay << "s" << endl;

    // Mettre à jour les statistiques
    numReceived++;
    packetsPerUAV[srcAddr]++;
    emit(rcvdPkSignal, packet);

    // Traiter le message FedAvg s'il en contient un
    auto chunk = packet->peekAtFront<Chunk>();
    if (auto fedAvgMsg = dynamicPtrCast<const FedAvgMessage>(chunk)) {
        processFedAvgMessage(fedAvgMsg.get()->dup(), srcAddr);
    }

    delete packet;
}

void BaseStationAppFedAvg::processFedAvgMessage(FedAvgMessage *msg, L3Address srcAddr) {
    if (msg->getMessageType() == LOCAL_UPDATE) {
        int uavId = msg->getUavId();
        int roundId = msg->getRoundId();

        if (roundId == currentRound) {
            EV_INFO << "Received model update from UAV " << uavId
                   << " for round " << roundId << endl;

            // Stocker le modèle reçu
            receivedModels[uavId] = msg->getModelWeights();
            samplesPerUav[uavId] = msg->getSamplesCount();

            // Émettre signal de précision si disponible
            double accuracy = msg->getAccuracy();
            if (accuracy > 0) {
                emit(modelAccuracySignal, accuracy);
            }

            // Si nous avons reçu les modèles de tous les UAVs, agréger
            if (static_cast<int>(receivedModels.size()) == numUavs) {
                EV_INFO << "Received models from all UAVs. Starting aggregation." << endl;
                aggregateModels();
            }
        }
        else {
            EV_WARN << "Received model update for round " << roundId
                   << " but current round is " << currentRound << endl;
        }
    }

    delete msg;
}

void BaseStationAppFedAvg::socketErrorArrived(UdpSocket *socket, Indication *indication) {
    EV_WARN << "Socket error: " << indication->getName() << endl;
    delete indication;
}

void BaseStationAppFedAvg::socketClosed(UdpSocket *socket) {
    if (operationalState == State::STOPPING_OPERATION) {
        startActiveOperationExtraTimeOrFinish(par("stopOperationExtraTime"));
    }
}

void BaseStationAppFedAvg::handleStartOperation(LifecycleOperation *operation) {
    socket.setOutputGate(gate("socketOut"));
    // Remove the duplicate socket binding from here
    socket.setCallback(this);

    roundTimer = new cMessage("roundTimer");
    scheduleAt(simTime() + par("startTime"), roundTimer);
}

void BaseStationAppFedAvg::handleStopOperation(LifecycleOperation *operation) {
    cancelEvent(roundTimer);
    socket.close();
    delayActiveOperationFinish(par("stopOperationTimeout"));
}

void BaseStationAppFedAvg::handleCrashOperation(LifecycleOperation *operation) {
    cancelEvent(roundTimer);
    socket.destroy();
}

void BaseStationAppFedAvg::finish() {
    ApplicationBase::finish();

    EV_INFO << "Base Station FedAvg Application finished." << endl;
    EV_INFO << "Completed " << currentRound << " rounds of federated learning." << endl;
    EV_INFO << "Received " << numReceived << " packets in total." << endl;
    EV_INFO << "Packets received from each UAV:" << endl;

    for (auto& pair : packetsPerUAV) {
        EV_INFO << "  UAV at " << pair.first.str() << ": " << pair.second << " packets" << endl;
    }
}
