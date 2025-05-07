import inet.applications.contract.IApp;

simple BaseStationAppFedAvg like IApp
{
    parameters:
        string interfaceTableModule;
        int localPort;
        int fedAvgPort = default(9000);          // Port pour la communication FedAvg
        int numUavs = default(5);                // Nombre d'UAVs dans le réseau
        int maxRounds = default(10);             // Nombre maximal de cycles d'apprentissage
        double roundInterval @unit(s) = default(20s); // Intervalle entre les rondes
        double startTime @unit(s) = default(5s); // Délai de démarrage
        double stopOperationExtraTime @unit(s) = default(2s);
        double stopOperationTimeout @unit(s) = default(2s);
        
        @display("i=block/app");
        @signal[rcvdPk](type=inet::Packet);
        @signal[roundCompleted](type=int);
        @signal[modelAccuracy](type=double);
        @statistic[rcvdPk](title="packets received"; source=rcvdPk; record=count,"sum(packetBytes)","vector(packetBytes)"; interpolationmode=none);
        @statistic[roundCompleted](title="rounds completed"; source=roundCompleted; record=vector);
        @statistic[modelAccuracy](title="model accuracy"; source=modelAccuracy; record=vector,stats);
        
    gates:
        input socketIn;
        output socketOut;
}