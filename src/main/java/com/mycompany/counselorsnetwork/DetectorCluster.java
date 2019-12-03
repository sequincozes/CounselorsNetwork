/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.counselorsnetwork;

import java.util.ArrayList;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;

/**
 *
 * @author silvio
 */
public class DetectorCluster {

    DetectorClassifier[] classifiers = {
        new DetectorClassifier(new RandomTree(), "Random Tree", "BENIGN"),
        new DetectorClassifier(new RandomForest(), "Random Forest", "BENIGN"),
        new DetectorClassifier(new NaiveBayes(), "Naive Bayes", "BENIGN"),
        new DetectorClassifier(new J48(), "J48", "BENIGN"),
        new DetectorClassifier(new REPTree(), "REP Tree", "BENIGN")};
    ArrayList<Integer> clusteredInstancesIndex; //[cluster][index]
    int clusterNum;
    double threshold = 2; // 2% do best 

    public DetectorCluster(int clusterNum) {
        this.clusteredInstancesIndex = new ArrayList<Integer>();
        this.clusterNum = clusterNum;
    }

    public void addInstanceIndex(int index) {
        this.clusteredInstancesIndex.add(index);
    }

    public void evaluateClassifiers(Instances dataEvaluation) throws Exception {
        for (DetectorClassifier c : classifiers) {
            c.resetAndEvaluate(dataEvaluation, clusteredInstancesIndex);
        }
    }

    public void classifierSelection() throws Exception {
        DetectorClassifier best = classifiers[0];
        for (DetectorClassifier c : classifiers) {
            if (c.evaluationAccuracy > best.getEvaluationAccuracy()) {
                best = c;
                System.out.println("Best: " + c.getName() + " (" + c.getEvaluationAccuracy() + ")");
            }
        }

        for (DetectorClassifier c : classifiers) {
            if (c.evaluationAccuracy + threshold >= best.getEvaluationAccuracy()) {
                c.setSelected(true);
            } else {
                c.setSelected(false);
            }
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

    public int getClusterNum() {
        return clusterNum;
    }

    public void setClusterNum(int clusterNum) {
        this.clusterNum = clusterNum;
    }

    public double getThreshold() {
        return threshold;
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

}
