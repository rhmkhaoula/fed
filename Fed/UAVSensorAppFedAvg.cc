#include "UAVSensorAppFedAvg.h"
#include "inet/common/ModuleAccess.h"
#include "inet/common/TimeTag_m.h"
#include "inet/common/packet/chunk/ByteCountChunk.h"
#include "inet/networklayer/common/L3AddressResolver.h"
#include "inet/transportlayer/contract/udp/UdpControlInfo_m.h"
#include "inet/networklayer/common/L3AddressTag_m.h"

Define_Module(UAVSensorAppFedAvg);

simsignal_t UAVSensorAppFedAvg::sentPkSignal = registerSignal("sentPk");
simsignal_t UAVSensorAppFedAvg::rcvdPkSignal = registerSignal("rcvdPk");

UAVSensorAppFedAvg::UAVSensorAppFedAvg() : localModel(5, 0.01, 32, 3) {
}

UAVSensorAppFedAvg::~UAVSensorAppFedAvg() {
    cancelAndDelete(sendTimer);
    cancelAndDelete(trainTimer);
}

void UAVSensorAppFedAvg::initialize(int stage) {
    ApplicationBase::initialize(stage);

    if (stage == INITSTAGE_LOCAL) {
        sendInterval = par("sendInterval");
        localPort = par("localPort");
        destPort = par("destPort");
        fedAvgPort = par("fedAvgPort");
        uavId = par("uavId");

        trainingCompletedSignal = registerSignal("trainingCompleted");
        localAccuracySignal = registerSignal("localAccuracy");

        numSent = 0;
        numReceived = 0;
        currentRound = 0;
        trainingInProgress = false;

        WATCH(numSent);
        WATCH(numReceived);
        WATCH(currentRound);
        WATCH(trainingInProgress);

        // Générer des données synthétiques pour l'entraînement
        generateSyntheticData();
    }
    else if (stage == INITSTAGE_APPLICATION_LAYER) {
        // Configuration du socket
        socket.setOutputGate(gate("socketOut"));
        socket.setCallback(this);
        
        // Only bind the socket here if localPort is specified
        if (localPort > 0) {
            socket.bind(localPort);
        }

        // Résolution de l'adresse de la station de base pour les données normales
        const char *destAddrs = par("destAddresses");
        cStringTokenizer tokenizer(destAddrs);
        const char *token;

        while ((token = tokenizer.nextToken()) != nullptr) {
            L3AddressResolver().tryResolve(token, destAddress);
            if (destAddress.isUnspecified())
                EV_ERROR << "Cannot resolve destination address: " << token << endl;
            baseStationAddress = destAddress; // Utiliser la même adresse pour FedAvg
            break;
        }

        // Configurer les timers
        sendTimer = new cMessage("sendTimer");
        trainTimer = new cMessage("trainTimer");

        if (!destAddress.isUnspecified() && operationalState == State::OPERATING) {
            scheduleAt(simTime() + par("startTime"), sendTimer);
        }

        EV_INFO << "UAV[" << uavId << "] Sensor App FedAvg initialized. Ready to participate in federated learning." << endl;
    }
}

void UAVSensorAppFedAvg::generateSyntheticData() {
    // Générer des données synthétiques pour l'entraînement
    // Dans un cas réel, cela serait remplacé par des données réelles collectées par l'UAV
    EV_INFO << "Generating synthetic training data for UAV " << uavId << endl;

    // Nombre d'échantillons de données, avec une légère variation par UAV
    int numSamples = 100 + (uavId * 20);

    // Générateur de nombres aléatoires
    std::mt19937 rng(uavId + 1000); // Utiliser uavId comme graine pour avoir des données différentes par UAV
    std::uniform_real_distribution<double> featureDist(-5.0, 5.0);
    std::normal_distribution<double> noiseDist(0.0, 0.5);

    // Générer des données selon un modèle linéaire simple y = w1*x1 + w2*x2 + ... + bruit
    std::vector<double> trueWeights = {0.5, -1.2, 0.8, 2.0, -0.7};
    double trueBias = 1.0;

    trainingData.clear();
    for (int i = 0; i < numSamples; i++) {
        // Générer des features
        std::vector<double> features;
        for (size_t j = 0; j < trueWeights.size(); j++) {
            features.push_back(featureDist(rng));
        }

        // Calculer la sortie avec le vrai modèle + bruit
        double output = trueBias;
        for (size_t j = 0; j < features.size(); j++) {
            output += features[j] * trueWeights[j];
        }
        output += noiseDist(rng); // Ajouter du bruit

        // Stocker la paire (features, output)
        trainingData.push_back(std::make_pair(features, output));
    }

    EV_INFO << "Generated " << numSamples << " training samples for UAV " << uavId << endl;
}

void UAVSensorAppFedAvg::handleMessageWhenUp(cMessage *msg) {
    if (msg->isSelfMessage()) {
        if (msg == sendTimer) {
            sendSensorData();
            scheduleAt(simTime() + sendInterval, sendTimer);
        }
        else if (msg == trainTimer) {
            trainLocalModel();
        }
    }
    else {
        socket.processMessage(msg);
    }
}

void UAVSensorAppFedAvg::collectSensorData() {
    // Simulation de la collecte de données d'un capteur
    // Ici nous pourrions lire des données d'un fichier externe, ou générer des données synthétiques
    EV_INFO << "UAV[" << uavId << "] collecting sensor data..." << endl;
    // Pour cet exemple, nous générons simplement des données aléatoires
}

void UAVSensorAppFedAvg::sendSensorData() {
    collectSensorData();

    char msgName[32];
    sprintf(msgName, "UAVSensorData-%d-%d", uavId, numSent);

    // Créer un paquet avec les données du capteur
    Packet *packet = new Packet(msgName);

    // Ajouter un tag de temps pour mesurer la latence
    auto creationTimeTag = packet->addTag<CreationTimeTag>();
    creationTimeTag->setCreationTime(simTime());

    // Simuler les données du capteur avec un payload de taille fixe
    const auto& payload = makeShared<ByteCountChunk>(B(par("messageLength")));
    packet->insertAtBack(payload);

    // Envoyer à la station de base
    socket.sendTo(packet, destAddress, destPort);

    numSent++;
    emit(sentPkSignal, packet);
}

void UAVSensorAppFedAvg::trainLocalModel() {
    if (trainingData.empty()) {
        EV_WARN << "UAV[" << uavId << "] has no training data available" << endl;
        return;
    }

    EV_INFO << "UAV[" << uavId << "] training local model for round " << currentRound << endl;

    // Entraîner le modèle local avec les données
    localModel.train(trainingData);

    // Évaluer le modèle pour obtenir une métrique de performance
    double accuracy = evaluateModel();

    // Émettre un signal de fin d'entraînement
    emit(trainingCompletedSignal, currentRound);
    emit(localAccuracySignal, accuracy);

    EV_INFO << "UAV[" << uavId << "] training completed with accuracy: " << accuracy << endl;

    // Envoyer les mises à jour du modèle à la station de base
    sendModelUpdate();

    trainingInProgress = false;
}

double UAVSensorAppFedAvg::evaluateModel() {
    // Évaluer le modèle sur un ensemble de validation
    // Dans cet exemple, nous utilisons simplement l'erreur moyenne sur les données d'entraînement
    double totalError = 0.0;

    for (const auto& sample : trainingData) {
        const auto& features = sample.first;
        double target = sample.second;

        double prediction = localModel.predict(features);
        double error = std::abs(prediction - target);
        totalError += error;
    }

    double avgError = totalError / trainingData.size();
    double accuracy = 1.0 / (1.0 + avgError); // Convertir l'erreur en une mesure de "précision"

    return accuracy;
}

void UAVSensorAppFedAvg::sendModelUpdate() {
    char msgName[32];
    sprintf(msgName, "ModelUpdate-UAV%d-Round%d", uavId, currentRound);

    // Créer un paquet pour le message FedAvg
    Packet *packet = new Packet(msgName);

    // Créer le message FedAvg
    const auto& fedAvgMsg = makeShared<FedAvgMessage>();
    fedAvgMsg->setMessageType(LOCAL_UPDATE);
    fedAvgMsg->setRoundId(currentRound);
    fedAvgMsg->setModelWeights(localModel.serialize().c_str());
    fedAvgMsg->setUavId(uavId);
    fedAvgMsg->setAccuracy(evaluateModel());
    fedAvgMsg->setSamplesCount(trainingData.size());

    // Ajouter le message au paquet
    packet->insertAtBack(fedAvgMsg);

    // Envoyer à la station de base
    socket.sendTo(packet, baseStationAddress, fedAvgPort);

    EV_INFO << "UAV[" << uavId << "] sent model update to base station for round " << currentRound << endl;
}

void UAVSensorAppFedAvg::socketDataArrived(UdpSocket *socket, Packet *packet) {
    // Traitement des paquets reçus
    auto addressInd = packet->getTag<L3AddressInd>();
    L3Address srcAddr = addressInd->getSrcAddress();

    EV_INFO << "UAV[" << uavId << "] received packet " << packet->getName()
            << " from " << srcAddr.str() << endl;

    numReceived++;
    emit(rcvdPkSignal, packet);

    // Vérifier si c'est un message FedAvg
    auto chunk = packet->peekAtFront<Chunk>();
    if (auto fedAvgMsg = dynamicPtrCast<const FedAvgMessage>(chunk)) {
        processFedAvgMessage(fedAvgMsg.get()->dup());
    }

    delete packet;
}

void UAVSensorAppFedAvg::processFedAvgMessage(FedAvgMessage *msg) {
    if (msg->getMessageType() == GLOBAL_UPDATE) {
        int roundId = msg->getRoundId();

        EV_INFO << "UAV[" << uavId << "] received global model update for round " << roundId << endl;

        // Mettre à jour notre ronde actuelle
        currentRound = roundId;

        // Mettre à jour notre modèle local avec le modèle global
        localModel.deserialize(msg->getModelWeights());

        // Planifier l'entraînement local
        if (!trainingInProgress) {
            trainingInProgress = true;
            // Ajouter un petit délai pour éviter que tous les UAVs s'entraînent exactement en même temps
            simtime_t trainDelay = 0.1 + 0.05 * uavId;
            scheduleAt(simTime() + trainDelay, trainTimer);

            EV_INFO << "UAV[" << uavId << "] scheduled local training in " << trainDelay << "s" << endl;
        }
    }

    delete msg;
}

void UAVSensorAppFedAvg::socketErrorArrived(UdpSocket *socket, Indication *indication) {
    EV_WARN << "Socket error: " << indication->getName() << endl;
    delete indication;
}

void UAVSensorAppFedAvg::socketClosed(UdpSocket *socket) {
    if (operationalState == State::STOPPING_OPERATION) {
        startActiveOperationExtraTimeOrFinish(par("stopOperationExtraTime"));
    }
}

void UAVSensorAppFedAvg::handleStartOperation(LifecycleOperation *operation) {
    socket.setOutputGate(gate("socketOut"));
    socket.setCallback(this);
    // Remove the duplicate socket binding from here
    // Only bind if localPort is valid
    if (localPort > 0 && !socket.isOpen()) {
        socket.bind(localPort);
    }

    if (!destAddress.isUnspecified()) {
        sendTimer = new cMessage("sendTimer");
        trainTimer = new cMessage("trainTimer");
        scheduleAt(simTime() + par("startTime"), sendTimer);
    }
}

void UAVSensorAppFedAvg::handleStopOperation(LifecycleOperation *operation) {
    cancelEvent(sendTimer);
    cancelEvent(trainTimer);
    socket.close();
    delayActiveOperationFinish(par("stopOperationTimeout"));
}

void UAVSensorAppFedAvg::handleCrashOperation(LifecycleOperation *operation) {
    cancelEvent(sendTimer);
    cancelEvent(trainTimer);
    socket.destroy();
}

void UAVSensorAppFedAvg::finish() {
    ApplicationBase::finish();
    EV_INFO << "UAV[" << uavId << "] Sensor App FedAvg finished. Sent: " << numSent
            << " packets, Received: " << numReceived << " packets." << endl;
    EV_INFO << "Participated in " << currentRound << " rounds of federated learning." << endl;
}
