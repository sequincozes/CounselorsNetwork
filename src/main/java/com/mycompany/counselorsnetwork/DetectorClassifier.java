/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.counselorsnetwork;

import java.util.ArrayList;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author silvio
 */
public class DetectorClassifier {

    Classifier classifier;
    String name;
    double evaluationAccuracy;
    double testAccuracy;
    int VP, VN, FP, FN;
    long evaluationNanotime;
    long testNanotime;
    long trainNanotime;

    public DetectorClassifier(Classifier classifier, String name) {
        this.classifier = classifier;
        this.name = name;
    }

    public Classifier train(Instances dataTrain) throws Exception {
        long currentTime = System.nanoTime();
        classifier.buildClassifier(dataTrain);
        setTrainNanotime(System.nanoTime() - currentTime);
        return classifier;
    }

    public Classifier resetAndClassify(Instances dataTest, String normalClassName, boolean evaluation, ArrayList<Integer> clusteredInstances) throws Exception {
        long currentTime = System.nanoTime();
        setVN(0);
        setVP(0);
        setFN(0);
        setFP(0);
        dataTest.setClassIndex(dataTest.numAttributes() - 1);
        for (int index = 0; index < clusteredInstances.size(); index++) {
            Instance instance = dataTest.get(index);
            if (classify(instance) == instance.classValue()) {
                if (instance.stringValue(instance.attribute(instance.classIndex())).equals(normalClassName)) {
                    setVN(getVN() + 1);
                } else {
                    setVP(getVP() + 1);
                }
                // VP ou VN
            } else {
                // FP ou FN
                if (instance.stringValue(instance.attribute(instance.classIndex())).equals(normalClassName)) {
                    setFP(getFP() + 1);
                } else {
                    setFN(getFN() + 1);
                }
            }
        }
        long classificationTime = System.nanoTime() - currentTime;
        double accuracy = Float.valueOf(((getVP() + getVN()) * 100) / (getVP() + getVN() + getFP() + getFN()));
//        this.taxaDeteccao = Float.valueOf((getVP() * 100) / (getVP() + getFN()));
//        this.taxaAlarmeFalsos = Float.valueOf((getFP() * 100) / (getVN() + getFP()));
        if (evaluation) {
            setEvaluationNanotime(classificationTime);
            setEvaluationAccuracy(accuracy);
        } else {
            setTestNanotime(classificationTime);
            setTestAccuracy(accuracy);
        }

        return classifier;
    }

    public double classify(Instance singleInstance) throws Exception {
//        System.out.println("Classificando: " + singleInstance);
        return this.classifier.classifyInstance(singleInstance);
    }

    public Classifier getClassifier() {
        return classifier;
    }

    public void setClassifier(Classifier classifier) {
        this.classifier = classifier;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public double getEvaluationAccuracy() {
        return evaluationAccuracy;
    }

    public void setEvaluationAccuracy(double evaluationAccuracy) {
        this.evaluationAccuracy = evaluationAccuracy;
    }

    public double getTestAccuracy() {
        return testAccuracy;
    }

    public void setTestAccuracy(double testAccuracy) {
        this.testAccuracy = testAccuracy;
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

    public long getEvaluationNanotime() {
        return evaluationNanotime;
    }

    public void setEvaluationNanotime(long evaluationNanotime) {
        this.evaluationNanotime = evaluationNanotime;
    }

    public long getTestNanotime() {
        return testNanotime;
    }

    public void setTestNanotime(long testNanotime) {
        this.testNanotime = testNanotime;
    }

    public long getTrainNanotime() {
        return trainNanotime;
    }

    public void setTrainNanotime(long trainNanotime) {
        this.trainNanotime = trainNanotime;
    }

}
