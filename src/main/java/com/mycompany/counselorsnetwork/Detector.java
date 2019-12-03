/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.counselorsnetwork;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;

/**
 *
 * @author silvio
 */
public class Detector {

    SimpleKMeans kmeans;
    DetectorCluster[] clusters;
    DetectorClassifier[] classifiers;
    Instances trainInstances;
    Instances evaluationInstances;
    Instances evaluationInstancesNoLabel;
    Instances testInstances;

    public Detector(DetectorClassifier[] classifiers, Instances trainInstances, Instances evaluationInstances, Instances testInstances) {
        this.classifiers = classifiers;
        this.trainInstances = trainInstances;
        this.evaluationInstances = evaluationInstances;
        this.evaluationInstancesNoLabel = new Instances(evaluationInstances);
        evaluationInstancesNoLabel.deleteAttributeAt(evaluationInstancesNoLabel.numAttributes() - 1);  // Removendo classe
        this.testInstances = testInstances;
    }

    public void createClusters(int k, int seed) throws Exception {
        clusters = new DetectorCluster[k];
        kmeans = new SimpleKMeans();
        kmeans.setSeed(seed);
        kmeans.setPreserveInstancesOrder(true);
        kmeans.setNumClusters(k);
        kmeans.buildClusterer(evaluationInstancesNoLabel);

        for (int ki = 0; ki < k; ki++) {
            clusters[ki] = new DetectorCluster(classifiers);
        }
        int[] assignments = kmeans.getAssignments(); // Avaliação No-Label
        for (int i = 0; i < assignments.length; i++) {
            int cluster = assignments[i];
            clusters[cluster].addInstanceIndex(i);
//             System.out.println("Cluster " + cluster + ": " + i);
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

    public DetectorCluster[] getClusters() {
        return clusters;
    }

    public void setClusters(DetectorCluster[] clusters) {
        this.clusters = clusters;
    }

    public DetectorClassifier[] getClassifiers() {
        return classifiers;
    }

    public void setClassifiers(DetectorClassifier[] classifiers) {
        this.classifiers = classifiers;
    }

}
