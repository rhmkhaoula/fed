// Fichier de configuration du réseau (UAVNetwork.ned)


import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.physicallayer.wireless.common.contract.packetlevel.IRadioMedium;
import inet.physicallayer.wireless.ieee80211.packetlevel.Ieee80211ScalarRadioMedium;
import inet.visualizer.contract.IIntegratedVisualizer;
import inet.node.inet.AdhocHost;
import inet.mobility.contract.IMobility;
import inet.mobility.single.CircleMobility;
import inet.mobility.static.StationaryMobility;

network UAVNetwork
{
    parameters:
        @display("bgb=800,600;bgi=background/terrain,s");
        
    submodules:
        visualizer: <default("IntegratedCanvasVisualizer")> like IIntegratedVisualizer {
            parameters:
                @display("p=100,50");
        }
        
        radioMedium: Ieee80211ScalarRadioMedium {
            parameters:
                @display("p=100,150");
        }
        
        configurator: Ipv4NetworkConfigurator {
            parameters:
                @display("p=100,250");
        }
        
        baseStation: AdhocHost {
            parameters:
                @display("p=400,500;i=device/antennatower");
        }
        
        uav[5]: AdhocHost {
            parameters:
                @display("i=misc/drone");
        }
    
    connections allowunconnected:
}