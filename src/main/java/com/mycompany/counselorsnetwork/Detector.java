/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.counselorsnetwork;

import java.util.ArrayList;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author silvio
 */
public class Detector {

    SimpleKMeans kmeans;
    DetectorCluster[] clusters;
    Instances trainInstances;
    Instances evaluationInstances;
    Instances evaluationInstancesNoLabel;
    Instances testInstances;
    int conflitos = 0;
    Instances testInstancesNoLabel;
    int VP, VN, FP, FN;
    String normalClass;

    ArrayList<Double> historicalData = new ArrayList<>();

    public Detector(Instances trainInstances, Instances evaluationInstances, Instances testInstances) {
        this.trainInstances = trainInstances;
        this.evaluationInstances = evaluationInstances;
        this.evaluationInstancesNoLabel = new Instances(evaluationInstances);
        evaluationInstancesNoLabel.deleteAttributeAt(evaluationInstancesNoLabel.numAttributes() - 1);  // Removendo classe
        this.testInstances = testInstances;
        this.testInstancesNoLabel = new Instances(testInstances);;
        testInstancesNoLabel.deleteAttributeAt(testInstancesNoLabel.numAttributes() - 1);  // Removendo classe
        testInstances.setClassIndex(evaluationInstances.numAttributes() - 1);
    }

    public void createClusters(int k, int seed) throws Exception {
        clusters = new DetectorCluster[k];
        kmeans = new SimpleKMeans();
        kmeans.setSeed(seed);
        kmeans.setPreserveInstancesOrder(true);
        kmeans.setNumClusters(k);
        kmeans.buildClusterer(evaluationInstancesNoLabel);

        for (int ki = 0; ki < k; ki++) {
            clusters[ki] = new DetectorCluster(ki);
        }
        int[] assignments = kmeans.getAssignments(); // Avaliação No-Label
        for (int i = 0; i < assignments.length; i++) {
            int cluster = assignments[i];
            clusters[cluster].addInstanceIndex(i);
        }

    }

    public void trainClassifiers() throws Exception {
        for (DetectorCluster cluster : clusters) {
            cluster.trainClassifiers(trainInstances);
        }
    }

    public void evaluateClassifiersPerCluster() throws Exception {
        for (DetectorCluster cluster : clusters) {
            evaluationInstances.setClassIndex(evaluationInstances.numAttributes() - 1);
            cluster.evaluateClassifiers(evaluationInstances);
        }
    }

    public void clusterAndTestSample() throws Exception {
        // Zerando contadores
        for (DetectorCluster cluster : clusters) {
            for (DetectorClassifier classifier : cluster.getClassifiers()) {
                classifier.resetConters();
                conflitos = 0;
            }
        }

        // Calculando teste
        int maxSizeTrain = 10000;
        for (int index = 0; index < testInstances.size(); index++) {
            int clusterNum = kmeans.clusterInstance(testInstancesNoLabel.get(index));
            double classFirstSelec = -1;
            for (DetectorClassifier c : clusters[clusterNum].getClassifiers()) {
                /* Se o classificador for selecionado, classificar com ele */
                if (c.isSelected()) {
                    double result = c.testSingle(testInstances.get(index));

                    /* Verifica se o classificador atual está em conflito com o
                        primeiro classificador selecionado */
                    if (classFirstSelec < 0) {
                        classFirstSelec = result;
                    } else {
                        Instance instance = testInstances.get(index);
                        if (c.testSingle(instance) != classFirstSelec) {
                            historicalData.add(-77.0);
                            conflitos = conflitos + 1;
                            break;
//                            trainInstances.add(testInstances.get(index));
                        } else {
                            /* Armazena resultado por consenso */
                            historicalData.add(index, result);
                            if (result == instance.classValue()) {
                                if (instance.stringValue(instance.attribute(instance.classIndex())).equals(normalClass)) {
                                    VN = VN + 1;
                                } else {
                                    VP = VP + 1;
                                }
                            } else {
                                if (instance.stringValue(instance.attribute(instance.classIndex())).equals(normalClass)) {
                                    FP = FP + 1;
                                } else {
                                    FN = FN + 1;
                                }
                            }
                        }
                    }
                    // Teste Alimentando Geral
                    if (maxSizeTrain > 0) {
                        trainInstances.add(testInstances.get(index));
                        maxSizeTrain--;
                    }
                }
            }
        }

    }

    public double getAdvice(int timestamp) {
        return historicalData.get(timestamp);
    }

    public DetectorCluster[] getClusters() {
        return clusters;
    }

    public void setClusters(DetectorCluster[] clusters) {
        this.clusters = clusters;
    }

    void selectClassifierPerCluster() throws Exception {
        for (DetectorCluster cluster : clusters) {
            cluster.classifierSelection();
        }
    }

    public int getConflitos() {
        return conflitos;
    }

    public void resetConters() {
        setVN(0);
        setVP(0);
        setFN(0);
        setFP(0);
    }

    public int getVP() {
        return VP;
    }

    public void setVP(int VP) {
        this.VP = VP;
    }

    public int getVN() {
        return VN;
    }

    public void setVN(int VN) {
        this.VN = VN;
    }

    public int getFP() {
        return FP;
    }

    public void setFP(int FP) {
        this.FP = FP;
    }

    public int getFN() {
        return FN;
    }

    public void setFN(int FN) {
        this.FN = FN;
    }
    
     public double getDetectionAccuracy() {
        try {
            return Float.valueOf(
                    Float.valueOf((getVP() + getVN()) * 100)
                    / Float.valueOf(getVP() + getVN() + getFP() + getFN()));
        } catch (ArithmeticException e) {
//            System.out.println(e.getLocalizedMessage());
        }
        return -1;
    }
}
