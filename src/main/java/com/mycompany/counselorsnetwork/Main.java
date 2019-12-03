/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.counselorsnetwork;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
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
public class Main {

//    String test_file = ataque + "test_95.csv";
//    String normal_file = ataque + "normal_test_95.csv";
    public static void main(String[] args) throws IOException, Exception {
        Main m = new Main();
        int[] featureSelection = new int[]{79, 40, 68, 13, 55};
        Instances trainInstances = m.leadAndFilter(false, "/home/silvio/datasets/CICIDS2017_RC/DETECTOR_UM_WEDNESDAY/10_train_files/compilado_train.csv", featureSelection);
        Instances evaluationInstances = m.leadAndFilter(false, "/home/silvio/datasets/CICIDS2017_RC/DETECTOR_UM_WEDNESDAY/10_evaluation_files/compilado_evaluation.csv", featureSelection);
        Instances testInstances = m.leadAndFilter(false, "/home/silvio/datasets/CICIDS2017_RC/DETECTOR_UM_WEDNESDAY/80_test_files/compilado_test_160.csv", featureSelection);
//        Instances trainInstances = m.leadAndFilter(false, "/home/silvio/datasets/CICIDS2017_RC/pequeno1.csv", featureSelection);
//        Instances evaluationInstances = m.leadAndFilter(false, "/home/silvio/datasets/CICIDS2017_RC/pequeno2.csv", featureSelection);
//        Instances testInstances = m.leadAndFilter(false, "/home/silvio/datasets/CICIDS2017_RC/pequeno3.csv", featureSelection);

        Detector D1 = new Detector(trainInstances, evaluationInstances, testInstances);

        /* Setup */
        D1.createClusters(4, 4);

        /* Train Phase*/
        D1.trainClassifiers();

        /* Evaluation Phase */
        D1.evaluateClassifiersPerCluster();
        D1.selectClassifierPerCluster();
        System.out.println("------------------------------------------------------------------------");
        System.out.println("  --  Evaluation");
        System.out.println("------------------------------------------------------------------------");
        for (DetectorCluster d : D1.getClusters()) {
            System.out.println("---- Cluster " + d.clusterNum + ":");
            for (DetectorClassifier c : d.getClassifiers()) {
                if (c.isSelected()) {
                    System.out.println("[X]" + c.getName()
                            + " - " + c.getEvaluationAccuracy()
                            + " (VP;VN;FP;FN) = "
                            + "("
                            + c.getVP()
                            + ";" + c.getVN()
                            + ";" + c.getFP()
                            + ";" + c.getFN()
                            + ")"
                    );
                } else {
                    System.out.println(c.getName()
                            + "[N] - " + c.getEvaluationAccuracy()
                            + " (VP;VN;FP;FN) = "
                            + "("
                            + c.getVP()
                            + ";" + c.getVN()
                            + ";" + c.getFP()
                            + ";" + c.getFN()
                            + ")"
                    );
                }
            }

        }

        /* Test Phase */
        System.out.println("------------------------------------------------------------------------");
        System.out.println("  --  Test");
        System.out.println("------------------------------------------------------------------------");
        D1.clusterAndTestSample();
        for (DetectorCluster d : D1.getClusters()) {
            System.out.println("---- Cluster " + d.clusterNum + ":");
            for (DetectorClassifier c : d.getClassifiers()) {
                if (c.isSelected()) {
                    System.out.println("[X]" + c.getName()
                            + " - " + c.getTestAccuracy()
                            + " (VP;VN;FP;FN) = "
                            + "("
                            + c.getVP()
                            + ";" + c.getVN()
                            + ";" + c.getFP()
                            + ";" + c.getFN()
                            + ")"
                    );
                }
            }

        }

    }

    public Instances leadAndFilter(boolean printSelection, String file, int[] featureSelection) throws Exception {
        Instances instances = new Instances(readDataFile(file));
        if (featureSelection.length > 0) {
            instances = applyFilterKeep(instances, featureSelection);
            if (printSelection) {
                System.out.println(Arrays.toString(featureSelection) + " - ");
            }
        }

        return instances;

    }

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    public static Instances applyFilterKeep(Instances instances, int[] fs) {
        Arrays.sort(fs);
        for (int i = instances.numAttributes() - 1; i > 0; i--) {
            if (instances.numAttributes() <= fs.length) {
                System.err.println("O número de features (" + instances.numAttributes() + ") precisa ser maior que o filtro (" + fs.length + ").");
                return instances;
            }
            boolean deletar = true;
            for (int j : fs) {
                if (i == j) {
                    deletar = false;
//                    System.out.println("Manter [" + i + "]:" + instances.attribute(i));;
                }
            }
            if (deletar) {
                instances.deleteAttributeAt(i - 1);
            }
        }
        return instances;
    }
}
