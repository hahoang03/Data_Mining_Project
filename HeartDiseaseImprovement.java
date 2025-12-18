import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.clusterers.SimpleKMeans;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;


import java.util.Random;

public class HeartDiseaseImprovement {

    public static void main(String[] args) throws Exception {

        // ====== LOAD DATASET ======
        DataSource source = new DataSource("data/heart_transformed.arff");
        Instances data = source.getDataSet();

        // --- Set class là cột cuối ---
        data.setClassIndex(data.numAttributes() - 1);

        // --- Tạo dataset chỉ chứa features numeric (bỏ class cuối) ---
        Instances features = new Instances(data);
        features.setClassIndex(-1); // bỏ class để PCA không chạy trên class
        features.deleteAttributeAt(data.classIndex()); // xóa cột class khỏi features

        // =====================================================================
        // 1) PCA – DIMENSIONALITY REDUCTION
        // =====================================================================
        PrincipalComponents pca = new PrincipalComponents();
        pca.setVarianceCovered(0.95);
        pca.setMaximumAttributes(5);
        pca.setInputFormat(features);

        Instances pcaFeatures = Filter.useFilter(features, pca);

        // --- Thêm lại cột class vào PCA dataset ---
        pcaFeatures.insertAttributeAt(data.classAttribute(), pcaFeatures.numAttributes());
        pcaFeatures.setClassIndex(pcaFeatures.numAttributes() - 1);

        for (int i = 0; i < pcaFeatures.numInstances(); i++) {
            pcaFeatures.instance(i).setClassValue(data.instance(i).classValue());
        }

        // =====================================================================
        // 2) K-MEANS CLUSTERING – ADD CLUSTER LABEL AS NEW FEATURE
        // =====================================================================
        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(3);
        kmeans.setPreserveInstancesOrder(true);
        kmeans.buildClusterer(features);

        int[] assignments = kmeans.getAssignments();

        Instances clusteredData = new Instances(data);
        clusteredData.insertAttributeAt(new weka.core.Attribute("cluster_label"), clusteredData.numAttributes());

        for (int i = 0; i < clusteredData.numInstances(); i++) {
            clusteredData.instance(i).setValue(clusteredData.numAttributes() - 1, assignments[i]);
        }
        clusteredData.setClassIndex(data.classIndex());

        // =====================================================================
        // 3) MODEL TRAINING + EVALUATION WITH RUNTIME
        // =====================================================================
        //randomForest rf = new RandomForest();
        SMO smo = new SMO();


        // --- PCA dataset ---
        long startPCA = System.currentTimeMillis();
        Evaluation pcaEval = new Evaluation(pcaFeatures);
        //pcaEval.crossValidateModel(rf, pcaFeatures, 10, new Random(1));
        pcaEval.crossValidateModel(smo, pcaFeatures, 10, new Random(1));
        long endPCA = System.currentTimeMillis();

        // --- Cluster dataset ---
        long startCluster = System.currentTimeMillis();
        Evaluation clusterEval = new Evaluation(clusteredData);
        //clusterEval.crossValidateModel(rf, clusteredData, 10, new Random(1));
        clusterEval.crossValidateModel(smo, clusteredData, 10, new Random(1));
        long endCluster = System.currentTimeMillis();

        // =====================================================================
        // 4) PRINT RESULTS
        // =====================================================================
        System.out.println("===== MODEL ON PCA DATA =====");
        System.out.println(pcaEval.toSummaryString());
        System.out.printf("Accuracy: %.4f\n", 1 - pcaEval.errorRate());
        System.out.printf("F1-score (weighted): %.4f\n", pcaEval.weightedFMeasure());
        System.out.println("Runtime: " + (endPCA - startPCA) + " ms\n");

        System.out.println("===== MODEL WITH KMEANS CLUSTER FEATURE =====");
        System.out.println(clusterEval.toSummaryString());
        System.out.printf("Accuracy: %.4f\n", 1 - clusterEval.errorRate());
        System.out.printf("F1-score (weighted): %.4f\n", clusterEval.weightedFMeasure());
        System.out.println("Runtime: " + (endCluster - startCluster) + " ms");
    }
}
