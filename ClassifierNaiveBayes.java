import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes; // Import Naive Bayes
import weka.classifiers.Evaluation;
import java.util.Random;

public class ClassifierNaiveBayes {

    public static void main(String[] args) {
        try {
            // 1. Load Dataset
            System.out.println("Loading data...");
            DataSource source = new DataSource("data/dataset.arff");
            Instances data = source.getDataSet();

            // Set class index to the last column (the target)
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // 2. Initialize Naive Bayes
            System.out.println("Initializing Naive Bayes...");
            // Naive Bayes is a probabilistic classifier based on Bayes' theorem
            NaiveBayes nb = new NaiveBayes();

            // 3. Train and Evaluate (10-Fold Cross Validation)
            System.out.println("Running Cross-Validation...");
            Evaluation eval = new Evaluation(data);
            
            // 10-fold CV with Seed 1
            eval.crossValidateModel(nb, data, 10, new Random(1));

            // 4. Output Results
            System.out.println("\n=== Naive Bayes Results ===");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString("=== Confusion Matrix ==="));
            System.out.printf("Accuracy: %.2f%%\n", eval.pctCorrect());
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}