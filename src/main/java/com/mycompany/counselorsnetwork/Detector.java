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
    Instances testInstancesNoLabel;
    int VP, VN, FP, FN;
    int conflitos = 0;
    int goodAdvices = 0;
    String normalClass;
    Detector[] advisors;
    boolean saveTrainInsance = true; // Ponteiro para dividir entre treino e validação
    ArrayList<Advice> historicalData = new ArrayList<>();
    String strTestAcc = "";

    public Detector(Instances trainInstances, Instances evaluationInstances, Instances testInstances, String normalClass) {
        this.trainInstances = trainInstances;
        this.evaluationInstances = evaluationInstances;
        this.evaluationInstancesNoLabel = new Instances(evaluationInstances);
        evaluationInstancesNoLabel.deleteAttributeAt(evaluationInstancesNoLabel.numAttributes() - 1);  // Removendo classe
        this.testInstances = testInstances;
        this.testInstancesNoLabel = new Instances(testInstances);
        testInstancesNoLabel.deleteAttributeAt(testInstancesNoLabel.numAttributes() - 1);  // Removendo classe
        testInstances.setClassIndex(evaluationInstances.numAttributes() - 1);
        advisors = new Detector[0];
        this.normalClass = normalClass;

    }

    public Detector(Instances trainInstances, Instances evaluationInstances, Instances testInstances, Detector[] advisors, String normalClass) {
        this.trainInstances = trainInstances;
        this.evaluationInstances = evaluationInstances;
        this.evaluationInstancesNoLabel = new Instances(evaluationInstances);
        evaluationInstancesNoLabel.deleteAttributeAt(evaluationInstancesNoLabel.numAttributes() - 1);  // Removendo classe
        this.testInstances = testInstances;
        this.testInstancesNoLabel = new Instances(testInstances);;
        testInstancesNoLabel.deleteAttributeAt(testInstancesNoLabel.numAttributes() - 1);  // Removendo classe
        testInstances.setClassIndex(evaluationInstances.numAttributes() - 1);
        this.advisors = advisors;
        this.normalClass = normalClass;
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

    public void trainClassifiers(boolean showTrainingTime) throws Exception {
        for (DetectorCluster cluster : clusters) {
            cluster.trainClassifiers(trainInstances, showTrainingTime);
        }
    }

    public void evaluateClassifiersPerCluster(boolean printEvaluation, boolean showProgress) throws Exception {
        for (DetectorCluster cluster : clusters) {
            evaluationInstances.setClassIndex(evaluationInstances.numAttributes() - 1);
            cluster.evaluateClassifiers(evaluationInstances);
        }
        selectClassifierPerCluster(showProgress);
        if (printEvaluation) {
            printEvaluationResults();
        }

    }

    public void clusterAndTestSample(boolean enableAdvice, boolean learnWithAdvice, boolean learnWithoutAdvices, boolean printEvaResults, boolean showProgress) throws Exception {
        // Calculando teste
//        int maxSizeTrain = 10000;
        boolean csv = true;
        int fimTrafegoNormal = -1;
        int percentRetrofeedAlfa = 10; //num of segments
        int percentRetrofeed = testInstances.size() / percentRetrofeedAlfa;
        int nextPoint = percentRetrofeed;
        System.out.println("Ten percent: " + percentRetrofeed);

        for (int instIndex = 0; instIndex < testInstances.size(); instIndex++) {
            // System.out.println("#############");
            // System.out.println("##  Testando - " + instIndex + "/" + testInstances.size());
            // System.out.println("#############");
            /* Instância Atual */
            Instance instance = testInstances.get(instIndex);
            double correctValue = instance.classValue();

            if (!instance.stringValue(instance.attribute(instance.classIndex())).equals(normalClass)) {
                if (fimTrafegoNormal == -1) {
                    fimTrafegoNormal = instIndex;
                }
            }

            if (nextPoint == instIndex) {
                nextPoint = nextPoint + percentRetrofeed;
                strTestAcc = strTestAcc + getDetectionAccuracyString() + ";";
                System.out.println("\n\n\n\nAcc;" + strTestAcc);// + ";" + trainInstances.size() + ";" + evaluationInstances.size());
                if (learnWithoutAdvices) {
                    trainClassifiers(false);
                    evaluateClassifiersPerCluster(printEvaResults, showProgress);
                }
                for (DetectorCluster cluster : clusters) {
                    cluster.printStrEvaluation();
                }
            }

            Instance evaluatingPeer = testInstancesNoLabel.get(instIndex);
            int clusterNum = kmeans.clusterInstance(evaluatingPeer);
            ArrayList<DetectorClassifier> selectedClassifiers = clusters[clusterNum].getSelectedClassifiers();
            int qtdClassificadores = selectedClassifiers.size();
            double classifiersOutput[][] = new double[qtdClassificadores][testInstances.size()];

            /* Se o classificador for selecionado, classificar com ele */
            if (qtdClassificadores == 0) {
                double adviceResult = handleConflict(enableAdvice, correctValue, instIndex, instance, learnWithAdvice, evaluatingPeer, printEvaResults, showProgress);
                System.out.println("[No Classifiers] Conflito n" + conflitos + " na instância " + instIndex + ", conselho: " + adviceResult + " / correto: " + correctValue);
            } else {
                for (int classifIndex = 0; classifIndex < qtdClassificadores; classifIndex++) {
                    DetectorClassifier c = selectedClassifiers.get(classifIndex);

                    double result = c.testSingle(instance);
                    classifiersOutput[classifIndex][instIndex] = result;

                    // Se nao for o primeiro classificador, mas houverem mais de um selecionado
                    if (classifIndex > 0 && classifIndex < qtdClassificadores - 1 && selectedClassifiers.size() > 1) {
                        // Checa conflito com o anterior
                        if (classifiersOutput[classifIndex][instIndex] != classifiersOutput[classifIndex - 1][instIndex]) {
                            double adviceResult = handleConflict(enableAdvice, correctValue, instIndex, instance, learnWithAdvice, evaluatingPeer, printEvaResults, showProgress);
                            System.out.println("[Divergence Classifiers] Conflito n" + conflitos + " na instância " + instIndex + ", conselho: " + adviceResult + " / correto: " + correctValue);
                        }
                        /* Se esse for o ultimo classificador da lista, é porque nao ocorreram conflitos*/
                    } else if (classifIndex == (qtdClassificadores - 1)) {
                        /* Salva um conselho para ser oferecido a outro detector */
                        historicalData.add(instIndex, new Advice(c.evaluationAccuracy, result, correctValue, normalClass));
                        updateResults(result, correctValue, instance);
                        /* Aprende sem Conflitos*/
                        if (learnWithoutAdvices) {
                            if (saveTrainInsance) {
                                trainInstances.add(instance); // Realimenta a cada amostra testada sem conflitos
                                saveTrainInsance = false;
//                            System.out.println("Aprendeu com conflito [" + conflitos + "].");
                            } else {
                                evaluationInstancesNoLabel.add(evaluatingPeer); // Realimenta a cada amostra testada sem conflitos
                                evaluationInstances.add(instance);
                                saveTrainInsance = true;
//                            System.out.println("Aprendeu com conflito.");
                            }
                        }
                    }

                }
            }

        }

        System.out.println(
                "Good Advices: " + getGoodAdvices() + "/" + conflitos);
        System.out.println(
                "Os ataques começaram na amostra " + fimTrafegoNormal + "(" + (fimTrafegoNormal / testInstances.numAttributes()) + "%)");
    }

    public Advice getAdvice(int timestamp) {
        return historicalData.get(timestamp);
    }

    public DetectorCluster[] getClusters() {
        return clusters;
    }

    public void setClusters(DetectorCluster[] clusters) {
        this.clusters = clusters;
    }

    void selectClassifierPerCluster(boolean showProgress) throws Exception {
        for (DetectorCluster cluster : clusters) {
            cluster.classifierSelection(showProgress);
        }
    }

    public int getConflitos() {
        return conflitos;
    }

    public void resetConters() {
        for (DetectorCluster cluster : clusters) {
            for (DetectorClassifier classifier : cluster.getClassifiers()) {
                classifier.resetConters();
            }
        }

        setVN(0);
        setVP(0);
        setFN(0);
        setFP(0);
        conflitos = 0;
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

    public String getDetectionAccuracyString() {
        try {
            double acc = Float.valueOf(
                    Float.valueOf((getVP() + getVN()) * 100)
                    / Float.valueOf(getVP() + getVN() + getFP() + getFN()));
            return String.valueOf(acc / 100).replace(".", ",");
        } catch (ArithmeticException e) {
//            System.out.println(e.getLocalizedMessage());
        }
        return "-1";
    }

    public int getCountTestInstances() {
        return testInstances.size();
    }

    public ArrayList<Advice> getHistoricalData() {
        return historicalData;
    }

    public int getGoodAdvices() {
        return goodAdvices;
    }

    private double handleConflict(boolean enableAdvice, double correctValue,
            int instIndex, Instance instance, boolean learnWithAdvice, Instance evaluatingPeer, boolean printEvaResu, boolean showProgress) throws Exception {
        historicalData.add(new Advice(0, -77, correctValue, normalClass));
        conflitos = conflitos + 1;

        double result = -77;
        /* REDE DE CONSELHOS */
        if (enableAdvice) {
//                            System.out.println("########## CONFLITO #########");
//                            System.out.println("Classifier output:" + classifiersOutput[classifIndex][instIndex] + " vs " + classifiersOutput[classifIndex - 1][instIndex]);
            if (advisors.length > 0) {
                for (Detector d : advisors) {
                    Advice advice = d.getAdvice(instIndex);
//                                    System.out.println("Requesting conseil...");
//                                    System.out.println("Advisors' response: " + advice.getAdvisorResult() + " (" + String.valueOf(advice.accuracy).substring(0, 5) + ")%, correct is: " + advice.correctResult + "(Good Adivices: " + getGoodAdvices() + ")");
                    result = advice.getAdvisorResult();
                    instance.setClassValue(result);
                    if (learnWithAdvice) {
                        trainInstances.add(instance); // Realimenta a cada conselho
                        /* Treina Todos Classificadores Novamente */
                        trainClassifiers(false);
                        evaluateClassifiersPerCluster(printEvaResu, showProgress);
                        selectClassifierPerCluster(showProgress);
                    }
                    if (advice.getAdvisorResult() == correctValue) {
                        goodAdvices = goodAdvices + 1;
//                                        System.out.println("Class: " + advice.getClassNormal());
                        if (advice.getClassNormal().equals(normalClass)) {
                            VN = VN + 1;
                        } else {
                            VP = VP + 1;
                        }
                    } else {
                        if (advice.getClassNormal().equals(normalClass)) {
                            FP = FP + 1;
                        } else {
                            FN = FN + 1;
                        }

                    }

                    if (learnWithAdvice) {
                        if (saveTrainInsance) {
                            trainInstances.add(instance); // Realimenta a cada amostra testada sem conflitos
                            saveTrainInsance = false;
//                                            System.out.println("Aprendeu com conflito [" + conflitos + "].");
                        } else {
                            evaluationInstances.add(instance);
                            evaluationInstancesNoLabel.add(evaluatingPeer); // Realimenta a cada amostra testada sem conflitos
                            saveTrainInsance = true;
//                                            System.out.println("Aprendeu com conflito [" + conflitos + "].");
                        }
                    }
                }
            } else {
                System.out.println("Ocorreu um conflito mas nao há conselheiros.");
            }

        }
        return result;
    }

    private void updateResults(double result, double correctValue, Instance instance) {
        if (result == correctValue) {
            if (instance.stringValue(instance.attribute(instance.classIndex())).equals(normalClass)) {
                VN = VN + 1;
            } else {
                VP = VP + 1;
            }
        } else {
            if (instance.stringValue(instance.attribute(instance.classIndex())).equals(normalClass)) {
                FP = FP + 1;
                instance.setClassValue(result);
            } else {
                FN = FN + 1;
                instance.setClassValue(result);
            }
        }
    }

    public void printEvaluationResults() {
        System.out.println("------------------------------------------------------------------------");
        System.out.println("  --  Evaluation");
        System.out.println("------------------------------------------------------------------------");
        for (DetectorCluster d : getClusters()) {
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

    public void printTestResults() {
        for (DetectorCluster d : getClusters()) {
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
                            + "/" + getCountTestInstances() + ")");
                }

            }

        }

        System.out.println("------------------------------------------------------------------------");
        System.out.println("  --  Test Summary: [Solucionados " + getGoodAdvices() + "/" + getConflitos() + " conflitos de " + (getVP() + getVN() + getFP() + getFN()) + " classificações.] \n "
                + "VP	VN	FP	FN	Acurácia \n"
                + getVP() + ";" + getVN() + ";" + getFP() + ";" + getFN() + ";" + String.valueOf(getDetectionAccuracy()).replace(".", ","));
        System.out.println("------------------------------------------------------------------------");
    }
}
