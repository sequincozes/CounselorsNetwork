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
import weka.core.Instances;

/**
 *
 * @author silvio
 */
public class Main {

//    String test_file = ataque + "test_95.csv";
//    String normal_file = ataque + "normal_test_95.csv";
    static final String NORMAL_CLASS = "BENIGN";

    public static void main(String[] args) throws IOException, Exception {
        Main m = new Main();
        int[] oneR_Detector1 = new int[]{79, 40, 68, 13, 55};
        int[] oneR_Detector2 = new int[]{79, 64, 5, 53, 35};

        /* Detector 1*/
        Instances trainInstances = m.leadAndFilter(false, "/home/silvio/datasets/CICIDS2017_RC/DETECTOR_UM_WEDNESDAY/10_train_files/compilado_train_100k.csv", oneR_Detector1);
        Instances evaluationInstances = m.leadAndFilter(false, "/home/silvio/datasets/CICIDS2017_RC/DETECTOR_UM_WEDNESDAY/10_evaluation_files/compilado_evaluation_100k.csv", oneR_Detector1);
        Instances testInstances = m.leadAndFilter(false, "/home/silvio/datasets/CICIDS2017_RC/DETECTOR_UM_WEDNESDAY/80_test_files/compilado_test_160_100k.csv", oneR_Detector1);

        /* Detector 2*/
        Instances trainInstances2 = m.leadAndFilter(false, "/home/silvio/datasets/CICIDS2017_RC/DETECTOR_DOIS_FRIDAY/10_train_files/compilado_train_10k.csv", oneR_Detector2);
        Instances evaluationInstances2 = m.leadAndFilter(false, "/home/silvio/datasets/CICIDS2017_RC/DETECTOR_DOIS_FRIDAY/10_evaluation_files/compilado_evaluation_100k.csv", oneR_Detector2);
        // Mesmo teste do D1, mas com outras features
        Instances testInstances2 = m.leadAndFilter(false, "/home/silvio/datasets/CICIDS2017_RC/DETECTOR_UM_WEDNESDAY/80_test_files/compilado_test_160_100k.csv", oneR_Detector2);


        /* Detector 1*/
        Detector D1 = new Detector(trainInstances, evaluationInstances, testInstances, NORMAL_CLASS);

        D1.createClusters(
                5, 2);
        System.out.println(
                "\n######## Detector 1");
        D1.resetConters();
        D1 = trainEvaluateAndTest(D1, false, false, false, true);

        /* Detector 1*/
        Detector[] advisors = {D1};
        Detector D2 = new Detector(trainInstances2, evaluationInstances2, testInstances2, advisors, NORMAL_CLASS);

        D2.createClusters(
                3, 2);
        System.out.println(
                "\n######## Detector 2");
        D2.resetConters();
        D2 = trainEvaluateAndTest(D2, false, false, true, true);

    }

    private static Detector trainEvaluateAndTest(Detector D1, boolean printEvaluation, boolean printTrain, boolean advices, boolean showProgress) throws Exception {
        /* Train Phase*/
        System.out.println("------------------------------------------------------------------------");
        System.out.println("  --  Train");
        System.out.println("------------------------------------------------------------------------");
        System.out.println("Treinamento com " + D1.trainInstances.numInstances() + " instâncias.");
        D1.trainClassifiers(printTrain);

        /* Evaluation Phase */
        D1.evaluateClassifiersPerCluster(printEvaluation, showProgress);       
                
        /* Test Phase */
        D1.clusterAndTestSample(advices, true, true,printEvaluation, showProgress);

//        int VP = 0;
//        int VN = 0;
//        int FP = 0;
//        int FN = 0;
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
                            + ") = ("
                            + (c.getVP() + c.getVN() + c.getFP() + c.getFN())
                            + "/" + D1.getCountTestInstances() + ")");
                    /* Atualiza Totais*/
                }

            }

        }
        System.out.println("------------------------------------------------------------------------");
        System.out.println("  --  Test Summary: [Solucionados "+ D1.getGoodAdvices()+"/"+ D1.getConflitos() + " conflitos de " + (D1.getVP() + D1.getVN() + D1.getFP() + D1.getFN()) + " classificações.] \n "
                + "VP	VN	FP	FN	Acurácia \n"
                + D1.getVP() + ";" + D1.getVN() + ";" + D1.getFP() + ";" + D1.getFN() + ";" + String.valueOf(D1.getDetectionAccuracy()).replace(".", ","));
        System.out.println("------------------------------------------------------------------------");

        return D1;
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
