#ifndef __FEDERATEDLEARNINGMODEL_H
#define __FEDERATEDLEARNINGMODEL_H

#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <random>
#include <memory>

/**
 * Classe représentant un modèle d'apprentissage fédéré simple.
 * Pour cette implémentation, nous utilisons un modèle de régression linéaire simple
 * comme exemple, mais cela pourrait être remplacé par un modèle plus complexe.
 */
class FederatedLearningModel {
protected:
    // Paramètres du modèle (poids)
    std::vector<double> weights;

    // Dimensions du modèle
    int inputDimension;

    // Hyperparamètres d'apprentissage
    double learningRate;
    int batchSize;
    int numEpochs;

    // Générateur de nombres aléatoires pour initialisation
    std::mt19937 rng;

public:
    /**
     * Constructeur
     * @param dimension Dimension d'entrée du modèle
     * @param lr Taux d'apprentissage
     * @param bSize Taille du lot pour l'entraînement
     * @param epochs Nombre d'époques d'entraînement
     */
    FederatedLearningModel(int dimension = 5, double lr = 0.01, int bSize = 32, int epochs = 3) :
        inputDimension(dimension),
        learningRate(lr),
        batchSize(bSize),
        numEpochs(epochs),
        rng(std::random_device()()) {

        // Initialiser les poids aléatoirement
        initializeWeights();
    }

    /**
     * Initialise les poids du modèle avec de petites valeurs aléatoires
     */
    void initializeWeights() {
        weights.resize(inputDimension + 1); // +1 pour le biais
        std::uniform_real_distribution<double> dist(-0.1, 0.1);

        for (int i = 0; i < weights.size(); i++) {
            weights[i] = dist(rng);
        }
    }

    /**
     * Prédit une valeur basée sur les entrées fournies
     * @param inputs Vecteur d'entrées
     * @return Valeur prédite
     */
    double predict(const std::vector<double>& inputs) {
        if (inputs.size() != inputDimension) {
            throw std::runtime_error("Dimension d'entrée incorrecte");
        }

        double result = weights[0]; // Biais
        for (int i = 0; i < inputDimension; i++) {
            result += inputs[i] * weights[i + 1];
        }

        return result;
    }

    /**
     * Entraîne le modèle sur un ensemble de données
     * @param data Ensemble de données (inputs, output)
     */
    void train(const std::vector<std::pair<std::vector<double>, double>>& data) {
        if (data.empty()) return;

        for (int epoch = 0; epoch < numEpochs; epoch++) {
            // Parcourir les données par lots
            for (int i = 0; i < data.size(); i += batchSize) {
                int batchEnd = std::min((int)data.size(), i + batchSize);

                // Calculer les gradients pour ce lot
                std::vector<double> gradients(weights.size(), 0.0);

                for (int j = i; j < batchEnd; j++) {
                    const auto& sample = data[j];
                    const auto& inputs = sample.first;
                    double target = sample.second;

                    // Prédiction
                    double prediction = predict(inputs);

                    // Erreur
                    double error = prediction - target;

                    // Mettre à jour le gradient du biais
                    gradients[0] += error;

                    // Mettre à jour les gradients des poids
                    for (int k = 0; k < inputDimension; k++) {
                        gradients[k + 1] += error * inputs[k];
                    }
                }

                // Normaliser les gradients par la taille du lot
                for (auto& grad : gradients) {
                    grad /= (batchEnd - i);
                }

                // Mettre à jour les poids
                for (int w = 0; w < weights.size(); w++) {
                    weights[w] -= learningRate * gradients[w];
                }
            }
        }
    }

    /**
     * Retourne les poids actuels du modèle
     * @return Vecteur de poids
     */
    const std::vector<double>& getWeights() const {
        return weights;
    }

    /**
     * Définit les poids du modèle
     * @param newWeights Nouveaux poids à définir
     */
    void setWeights(const std::vector<double>& newWeights) {
        if (newWeights.size() != weights.size()) {
            throw std::runtime_error("Dimension des poids incorrecte");
        }
        weights = newWeights;
    }

    /**
     * Sérialise le modèle en chaîne de caractères pour la transmission
     * @return Chaîne représentant les poids du modèle
     */
    std::string serialize() const {
        std::string result;
        for (const auto& w : weights) {
            if (!result.empty()) result += ";";
            result += std::to_string(w);
        }
        return result;
    }

    /**
     * Désérialise une chaîne en poids de modèle
     * @param serialized Chaîne sérialisée
     * @return true si la désérialisation a réussi
     */
    bool deserialize(const std::string& serialized) {
        std::vector<double> newWeights;
        std::string token;
        std::istringstream tokenStream(serialized);

        while (std::getline(tokenStream, token, ';')) {
            try {
                newWeights.push_back(std::stod(token));
            } catch (const std::exception& e) {
                return false;
            }
        }

        if (newWeights.size() == weights.size()) {
            weights = newWeights;
            return true;
        }

        return false;
    }
};

#endif
