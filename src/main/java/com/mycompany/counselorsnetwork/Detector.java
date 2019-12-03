/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.counselorsnetwork;

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
            }
        }

        // Calculando teste
        for (int index = 0; index < testInstances.size(); index++) {
            int clusterNum = kmeans.clusterInstance(testInstancesNoLabel.get(index));
            for (DetectorClassifier c : clusters[clusterNum].getClassifiers()) {
                if (c.isSelected()) {
//                    c.classifySingle(testInstances.get(index));
                }
            }
        }

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

}
