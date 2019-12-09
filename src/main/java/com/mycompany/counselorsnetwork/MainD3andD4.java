/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.counselorsnetwork;

import com.mycompany.counselorsnetwork.util.Util;
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
public class MainD3andD4 {

//    String test_file = ataque + "test_95.csv";
//    String normal_file = ataque + "normal_test_95.csv";
    static final String NORMAL_CLASS = "BENIGN";

    public static void main(String[] args) throws IOException, Exception {
        MainD3andD4 m = new MainD3andD4();
        int[] commonFilter = new int[]{79, 76, 70, 68, 67, 7, 6, 1};
        int[] D3Filter = new int[]{79, 70, 68, 7, 1};
        int[] D4Filter = new int[]{79, 76, 67, 6, 1};
        double fator = 20; // % test vs train

        String benign = "/home/silvio/datasets/CICIDS2017_RC/detector3/benign.csv";

        /* Detector 3*/
        System.out.println("\n######## Detector 3 (Web)");
        String brute_force_file = "/home/silvio/datasets/CICIDS2017_RC/detector3/brute_force.csv";
        String ftp_patator_file = "/home/silvio/datasets/CICIDS2017_RC/detector3/ftp_patator.csv";
        String sql_injection_file = "/home/silvio/datasets/CICIDS2017_RC/detector3/sql_injection.csv";
        String ssh_patator = "/home/silvio/datasets/CICIDS2017_RC/detector3/ssh_patator.csv";
        String xss_injection = "/home/silvio/datasets/CICIDS2017_RC/detector3/xss_injection.csv";
        Instances[] D3Instances = m.buildInstances(fator, new String[]{benign, brute_force_file, ftp_patator_file, sql_injection_file, ssh_patator, xss_injection}, D3Filter);
        System.out.println(D3Instances[0].size() + "/" + D3Instances[1].size() + "/" + D3Instances[2].size());
//        if (false) {
//            System.out.println("Features p/ Detector 3:");
//            Util.avaliarESelecionar(5, D3Instances[2], Util.METODO.GR, true);
//            System.exit(0);
        Detector D3 = new Detector(D3Instances[0], D3Instances[1], D3Instances[2], NORMAL_CLASS);

        D3.createClusters(4, 4);
        D3.resetConters();
        D3 = trainEvaluateAndTest(D3, false, false, false, true);
//        }

        /* Detector 4*/
//        if (true) {
        System.out.println("\n######## Detector 4 (DoS)");
        String goldeneye = "/home/silvio/datasets/CICIDS2017_RC/detector4/goldeneye.csv";
        String heartbleed = "/home/silvio/datasets/CICIDS2017_RC/detector4/heartbleed.csv";
        String hulk = "/home/silvio/datasets/CICIDS2017_RC/detector4/hulk.csv";
        String slowhttptest = "/home/silvio/datasets/CICIDS2017_RC/detector4/slowhttptest.csv";
        String slowloris = "/home/silvio/datasets/CICIDS2017_RC/detector4/slowloris.csv";
        Instances[] D4Instances = m.buildInstances(fator, new String[]{benign, goldeneye, heartbleed, hulk, slowhttptest, slowloris}, D4Filter);
        Instances[] D3InstancesGAMBI = m.buildInstances(fator, new String[]{benign, brute_force_file, ftp_patator_file, sql_injection_file, ssh_patator, xss_injection}, D4Filter);
//            System.out.println("Features p/ Detector 4:");
//            Util.avaliarESelecionar(15, D4Instances[2], Util.METODO.GR, true);
//            System.exit(0);
        Detector[] advisors = {D3};
        Detector D4 = new Detector(D4Instances[0], D4Instances[1], D3InstancesGAMBI[2], advisors, NORMAL_CLASS);
        D4.createClusters(6, 4);
        D4.resetConters();

        D4 = trainEvaluateAndTest(D4, false, false, true, true);
//        }
    }

    private static Detector trainEvaluateAndTest(Detector detectorTesting, boolean printEvaluation, boolean printTrain, boolean advices, boolean alwaysLearn) throws Exception {
        /* Train Phase*/
        System.out.println("------------------------------------------------------------------------");
        System.out.println("  --  Train");
        System.out.println("------------------------------------------------------------------------");
        System.out.println("Treinamento com " + detectorTesting.trainInstances.numInstances() + " instâncias.");
        detectorTesting.trainClassifiers(printTrain);

        /* Evaluation Phase */
        detectorTesting.evaluateClassifiersPerCluster();
        detectorTesting.selectClassifierPerCluster();
        if (printEvaluation) {
            System.out.println("------------------------------------------------------------------------");
            System.out.println("  --  Evaluation");
            System.out.println("------------------------------------------------------------------------");
            for (DetectorCluster d : detectorTesting.getClusters()) {
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
        }

        /* Test Phase */
        System.out.println("------------------------------------------------------------------------");
        System.out.println("  --  Test");
        System.out.println("------------------------------------------------------------------------");
        detectorTesting.resetConters();
        detectorTesting.clusterAndTestSample(advices, true, alwaysLearn);

//        int VP = 0;
//        int VN = 0;
//        int FP = 0;
//        int FN = 0;
        for (DetectorCluster d : detectorTesting.getClusters()) {
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
                            + "/" + detectorTesting.getCountTestInstances() + ")");
                    /* Atualiza Totais*/
                }

            }

        }
        System.out.println("------------------------------------------------------------------------");
        System.out.println("  --  Test Summary: [Solucionados " + detectorTesting.getGoodAdvices() + "/" + detectorTesting.getConflitos() + " conflitos de " + (detectorTesting.getVP() + detectorTesting.getVN() + detectorTesting.getFP() + detectorTesting.getFN()) + " classificações.] \n "
                + "VP	VN	FP	FN	Acurácia \n"
                + detectorTesting.getVP() + ";" + detectorTesting.getVN() + ";" + detectorTesting.getFP() + ";" + detectorTesting.getFN() + ";" + String.valueOf(detectorTesting.getDetectionAccuracy()).replace(".", ","));
        System.out.println("------------------------------------------------------------------------");

        return detectorTesting;
    }

    private Instances[] buildInstances(double factor, String[] instancesLocation, int[] filter) throws Exception {
        Instances train = null;
        Instances evaluation = null;
        Instances test = null;

        for (String local : instancesLocation) {
            Instances all = leadAndFilter(false, local, filter);
            Instances[] splitted50 = splitInstance(all, factor);
            Instances[] splitted25 = splitInstance(splitted50[0], factor);
            try {
                train.addAll(splitted25[0]);
                evaluation.addAll(splitted25[1]);
                test.addAll(splitted50[1]);
            } catch (NullPointerException e) {
                train = splitted25[0];
                evaluation = splitted25[1];
                test = splitted50[1];
            }
        }
        Instances[] splitted = {train, evaluation, test};
        return splitted;
    }

    public Instances leadAndFilter(boolean printSelection, String file, int[] featureSelection) throws Exception {
        Instances instances = new Instances(readDataFile(file));
        if (featureSelection.length > 0) {
            instances = applyFilterKeep(instances, featureSelection);
            if (printSelection) {
                System.out.println(Arrays.toString(featureSelection) + " - ");
            }
        } else {
            if (printSelection) {
                System.out.println("Sem filtro.");
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
                if (i != 1) {
                    System.err.println("O número de features (" + instances.numAttributes() + ") precisa ser maior que o filtro (" + fs.length + "). [i=" + i + "]");
                }
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

    public Instances[] splitInstance(Instances fullInstanceSet, double fator) {
        int trainSize = (int) Math.round(fullInstanceSet.numInstances() * fator / 100);
        int testSize = fullInstanceSet.numInstances() - trainSize;
        Instances train = new Instances(fullInstanceSet, 0, trainSize);
        Instances test = new Instances(fullInstanceSet, trainSize, testSize);
        Instances splitted[] = {train, test};
        return splitted;
    }
}
