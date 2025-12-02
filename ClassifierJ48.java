import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48; // Import J48 (C4.5 Decision Tree)
import weka.classifiers.Evaluation;
import java.util.Random;

public class ClassifierJ48 {

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

            // 2. Initialize J48
            System.out.println("Initializing J48 Decision Tree...");
            // J48 is Weka's implementation of the C4.5 decision tree algorithm
            J48 tree = new J48();
            // Optional: tree.setUnpruned(true); // If you want an unpruned tree

            // 3. Train and Evaluate (10-Fold Cross Validation)
            System.out.println("Running Cross-Validation...");
            Evaluation eval = new Evaluation(data);
            
            // 10-fold CV with Seed 1
            eval.crossValidateModel(tree, data, 10, new Random(1));

            // 4. Output Results
            System.out.println("\n=== J48 Decision Tree Results ===");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString("=== Confusion Matrix ==="));
            System.out.printf("Accuracy: %.2f%%\n", eval.pctCorrect());
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}