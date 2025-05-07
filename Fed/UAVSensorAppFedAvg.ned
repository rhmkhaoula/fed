import inet.applications.contract.IApp;

simple UAVSensorAppFedAvg like IApp
{
    parameters:
        string interfaceTableModule;
        double startTime @unit(s) = default(1s);
        double sendInterval @unit(s) = default(1s);
        int localPort = default(-1);
        int destPort;
        int fedAvgPort = default(9000);          // Port pour la communication FedAvg
        int uavId;                              // ID de l'UAV dans le réseau
        int messageLength @unit(B) = default(100B);
        string destAddresses = default("");
        double stopOperationExtraTime @unit(s) = default(2s);
        double stopOperationTimeout @unit(s) = default(2s);
        
        @display("i=block/app");
        @signal[sentPk](type=inet::Packet);
        @signal[rcvdPk](type=inet::Packet);
        @signal[trainingCompleted](type=int);
        @signal[localAccuracy](type=double);
        @statistic[sentPk](title="packets sent"; source=sentPk; record=count,"sum(packetBytes)","vector(packetBytes)"; interpolationmode=none);
        @statistic[rcvdPk](title="packets received"; source=rcvdPk; record=count,"sum(packetBytes)","vector(packetBytes)"; interpolationmode=none);
        @statistic[trainingCompleted](title="training rounds completed"; source=trainingCompleted; record=vector);
        @statistic[localAccuracy](title="local model accuracy"; source=localAccuracy; record=vector,stats);
        
    gates:
        input socketIn;
        output socketOut;
}