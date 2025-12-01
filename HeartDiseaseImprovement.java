import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.clusterers.SimpleKMeans;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.Logistic;
import weka.classifiers.Evaluation;

import java.util.Random;

public class HeartDiseaseImprovement {

    public static void main(String[] args) throws Exception {

        // ====== LOAD DATASET ======
        DataSource source = new DataSource("heart_disease.arff");  // đổi file bạn vào
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1); // Assume last column is target


        // =====================================================================
        // 1) PCA – DIMENSIONALITY REDUCTION
        // =====================================================================
        PrincipalComponents pca = new PrincipalComponents();
        pca.setVarianceCovered(0.95);  // giữ 95% phương sai
        pca.setMaximumAttributes(5);   // giảm xuống khoảng 5 features
        pca.setInputFormat(data);

        Instances pcaData = Filter.useFilter(data, pca);
        pcaData.setClassIndex(pcaData.numAttributes() - 1);


        // =====================================================================
        // 2) K-MEANS CLUSTERING – ADD CLUSTER LABEL AS NEW FEATURE
        // =====================================================================
        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(3);
        kmeans.buildClusterer(data);

        // Thêm cluster label vào data
        Instances clusteredData = new Instances(data);
        int[] assignments = kmeans.getAssignments();

        // thêm thuộc tính cluster
        clusteredData.insertAttributeAt(new weka.core.Attribute("cluster_label"), clusteredData.numAttributes());

        for (int i = 0; i < clusteredData.numInstances(); i++) {
            clusteredData.instance(i).setValue(clusteredData.numAttributes() - 1, assignments[i]);
        }

        clusteredData.setClassIndex(clusteredData.numAttributes() - 2); // class trước cluster


        // =====================================================================
        // 3) MODEL TRAINING
        // =====================================================================
        RandomForest baseRF = new RandomForest();
        Logistic baseLR = new Logistic();

        RandomForest improvedRF = new RandomForest();
        Logistic improvedLR = new Logistic();


        // --- Evaluate baseline on original data ---
        Evaluation baseEval = new Evaluation(data);
        baseEval.crossValidateModel(baseRF, data, 10, new Random(1));

        // --- Evaluate improved model on PCA data ---
        Evaluation pcaEval = new Evaluation(pcaData);
        pcaEval.crossValidateModel(improvedRF, pcaData, 10, new Random(1));

        // --- Evaluate improved model on KMeans-enhanced data ---
        Evaluation clusterEval = new Evaluation(clusteredData);
        clusterEval.crossValidateModel(improvedRF, clusteredData, 10, new Random(1));


        // =====================================================================
        // 4) PRINT RESULTS
        // =====================================================================
        System.out.println("===== BASE MODEL (RandomForest) ON ORIGINAL DATA =====");
        System.out.println(baseEval.toSummaryString());

        System.out.println("===== IMPROVED MODEL ON PCA DATA =====");
        System.out.println(pcaEval.toSummaryString());

        System.out.println("===== IMPROVED MODEL WITH KMEANS CLUSTER FEATURE =====");
        System.out.println(clusterEval.toSummaryString());
    }
}
