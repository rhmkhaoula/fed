// FedAvgMessage.msg
//
// Message utilisé pour la communication dans l'apprentissage fédéré
//

import inet.common.INETDefs;
import inet.common.packet.chunk.Chunk;

cplusplus {{
#include "inet/common/packet/chunk/FieldsChunk.h"
}}

namespace inet;

enum FedAvgMessageType {
    INIT_MODEL = 0;            // Message initial avec modèle global
    LOCAL_UPDATE = 1;          // Mise à jour locale d'un UAV
    GLOBAL_UPDATE = 2;         // Mise à jour globale de la station de base
    TRAINING_ROUND_START = 3;  // Début d'une ronde d'entraînement
    AGGREGATION_COMPLETE = 4;  // Agrégation des modèles terminée
};

class FedAvgMessage extends FieldsChunk {
    int messageType @enum(FedAvgMessageType);  // Type de message
    int roundId;                               // Identifiant de la ronde d'entraînement
    string modelWeights;                       // Poids du modèle sérialisés
    int uavId = -1;                            // ID de l'UAV (-1 pour station de base)
    double accuracy = 0.0;                     // Précision du modèle (optionnel)
    int samplesCount = 0;                      // Nombre d'échantillons utilisés pour l'entraînement
};
