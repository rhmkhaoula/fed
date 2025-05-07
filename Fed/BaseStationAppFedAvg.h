#ifndef __BASESTATIONAPPFEDAVG_H
#define __BASESTATIONAPPFEDAVG_H

#include <omnetpp.h>
#include "inet/applications/base/ApplicationBase.h"
#include "inet/transportlayer/contract/udp/UdpSocket.h"
#include "inet/common/lifecycle/LifecycleOperation.h"
#include "inet/common/packet/Packet.h"
#include "FederatedLearningModel.h"
#include "FedAvgMessage_m.h"

using namespace omnetpp;
using namespace inet;

/**
 * Application de la station de base implémentant l'algorithme FedAvg
 */
class BaseStationAppFedAvg : public ApplicationBase, public UdpSocket::ICallback {
  protected:
    // Configuration
    int localPort = -1;
    int numUavs = 0;           // Nombre total d'UAVs dans le réseau
    int fedAvgPort = 9000;     // Port dédié à la communication FedAvg
    int maxRounds = 10;        // Nombre maximal de cycles d'apprentissage fédéré

    // État
    UdpSocket socket;
    cMessage *roundTimer = nullptr;  // Timer pour démarrer chaque ronde
    simtime_t roundInterval;         // Intervalle entre les rondes

    // État FedAvg
    int currentRound = 0;
    std::map<int, std::string> receivedModels;  // Modèles reçus des UAVs
    std::map<int, int> samplesPerUav;           // Échantillons par UAV
    FederatedLearningModel globalModel;         // Modèle global

    // Statistiques
    int numReceived = 0;
    std::map<L3Address, int> packetsPerUAV;
    static simsignal_t rcvdPkSignal;
    simsignal_t roundCompletedSignal;
    simsignal_t modelAccuracySignal;

  protected:
    virtual void initialize(int stage) override;
    virtual void handleMessageWhenUp(cMessage *msg) override;
    virtual void finish() override;

    // Méthodes FedAvg
    virtual void startNextRound();
    virtual void aggregateModels();
    virtual void broadcastGlobalModel();

    // Méthodes d'application
    virtual void processPacket(Packet *pk);
    virtual void processFedAvgMessage(FedAvgMessage *msg, L3Address srcAddr);

    // Méthodes du socket
    virtual void socketDataArrived(UdpSocket *socket, Packet *packet) override;
    virtual void socketErrorArrived(UdpSocket *socket, Indication *indication) override;
    virtual void socketClosed(UdpSocket *socket) override;

    // LifecycleOperation
    virtual void handleStartOperation(LifecycleOperation *operation) override;
    virtual void handleStopOperation(LifecycleOperation *operation) override;
    virtual void handleCrashOperation(LifecycleOperation *operation) override;

  public:
    BaseStationAppFedAvg();
    virtual ~BaseStationAppFedAvg();
};

#endif
