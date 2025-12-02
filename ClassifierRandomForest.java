import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import java.util.Random;

public class ClassifierRandomForest {

    public static void main(String[] args) {
        try {
            // 1. load dataset
            System.out.println("Loading data...");
            DataSource source = new DataSource("data/dataset.arff");
            Instances data = source.getDataSet();

            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // 2. khoi tai random forest
            System.out.println("Initializing standard Random Forest...");
            RandomForest rf = new RandomForest();

            // 3. train va 10fold 
            System.out.println("Running Cross-Validation...");
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(rf, data, 10, new Random(1));

            // 4. output
            System.out.println("\n=== Random Forest Results ===");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString("=== Confusion Matrix ==="));
            System.out.printf("Accuracy: %.2f%%\n", eval.pctCorrect());
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}