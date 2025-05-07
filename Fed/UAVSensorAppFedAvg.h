#ifndef __UAVSENSORAPPFEDAVG_H
#define __UAVSENSORAPPFEDAVG_H

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
 * Application UAV implémentant l'algorithme d'apprentissage fédéré (FedAvg)
 */
class UAVSensorAppFedAvg : public ApplicationBase, public UdpSocket::ICallback {
  protected:
    // Configuration
    int localPort = -1;
    int destPort = -1;
    int fedAvgPort = 9000;     // Port dédié à la communication FedAvg
    L3Address destAddress;     // Adresse de la station de base
    L3Address baseStationAddress; // Adresse de la station de base pour FedAvg

    // État
    UdpSocket socket;
    cMessage *sendTimer = nullptr;
    cMessage *trainTimer = nullptr;
    simtime_t sendInterval;

    // État FedAvg
    int uavId;                // ID de l'UAV dans le réseau
    int currentRound = 0;     // Ronde d'apprentissage actuelle
    bool trainingInProgress = false;
    FederatedLearningModel localModel;  // Modèle local

    // Données synthétiques pour l'entraînement
    std::vector<std::pair<std::vector<double>, double>> trainingData;

    // Statistiques
    int numSent = 0;
    int numReceived = 0;
    static simsignal_t sentPkSignal;
    static simsignal_t rcvdPkSignal;
    simsignal_t trainingCompletedSignal;
    simsignal_t localAccuracySignal;

  protected:
    virtual void initialize(int stage) override;
    virtual void handleMessageWhenUp(cMessage *msg) override;
    virtual void finish() override;

    // Méthodes d'application
    virtual void sendSensorData();
    virtual void collectSensorData();

    // Méthodes FedAvg
    virtual void trainLocalModel();
    virtual void sendModelUpdate();
    virtual void generateSyntheticData();
    virtual double evaluateModel();

    // Méthodes de traitement des messages
    virtual void processFedAvgMessage(FedAvgMessage *msg);

    // Méthodes du socket
    virtual void socketDataArrived(UdpSocket *socket, Packet *packet) override;
    virtual void socketErrorArrived(UdpSocket *socket, Indication *indication) override;
    virtual void socketClosed(UdpSocket *socket) override;

    // LifecycleOperation
    virtual void handleStartOperation(LifecycleOperation *operation) override;
    virtual void handleStopOperation(LifecycleOperation *operation) override;
    virtual void handleCrashOperation(LifecycleOperation *operation) override;

  public:
    UAVSensorAppFedAvg();
    virtual ~UAVSensorAppFedAvg();
};

#endif
