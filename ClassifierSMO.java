import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.SMO; // Import SMO (SVM)
import weka.classifiers.Evaluation;
import java.util.Random;

public class ClassifierSMO {

    public static void main(String[] args) {
        try {
            // 1. load dataset
            System.out.println("Loading data...");
            DataSource source = new DataSource("data/dataset.arff");
            Instances data = source.getDataSet();
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // 2. smo
            System.out.println("Initializing SMO (Support Vector Machine)...");
            // SMO is Weka's implementation of SVM (Sequential Minimal Optimization)
            SMO smo = new SMO();

            // 3. train + 10fold cross valid
            System.out.println("Running Cross-Validation...");
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(smo, data, 10, new Random(1));

            // 4. output
            System.out.println("\n=== SMO Results ===");
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toMatrixString("=== Confusion Matrix ==="));
            System.out.printf("Accuracy: %.2f%%\n", eval.pctCorrect());
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}