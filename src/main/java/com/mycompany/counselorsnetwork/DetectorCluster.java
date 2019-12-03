/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.counselorsnetwork;

import java.util.ArrayList;
import weka.core.Instances;

/**
 *
 * @author silvio
 */
public class DetectorCluster {

    DetectorClassifier[] classifiers;
    ArrayList<Integer> clusteredInstancesIndex; //[cluster][index]

    public DetectorCluster(DetectorClassifier[] classifiers) {
        this.classifiers = classifiers;
        this.clusteredInstancesIndex = new ArrayList<>();
    }

    public void addInstanceIndex(int index) {
        this.clusteredInstancesIndex.add(index);
    }

    public void evaluateClassifiers(Instances dataEvaluation) throws Exception {
        for (DetectorClassifier c : classifiers) {
            c.resetAndClassify(dataEvaluation, "Normal", true, clusteredInstancesIndex);
        }
    }

    public void trainClassifiers(Instances dataTrain) throws Exception {
        for (DetectorClassifier c : classifiers) {
            dataTrain.setClassIndex(dataTrain.numAttributes() - 1);
            c.train(dataTrain);
        }
    }

    public DetectorClassifier[] getClassifiers() {
        return this.classifiers;
    }

    public ArrayList<Integer> getClusteredInstancesIndex() {
        return clusteredInstancesIndex;
    }

}
