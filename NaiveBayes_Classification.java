import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import java.util.Random;

public class NaiveBayes_Classification {
    public static void main(String[] args) {
        try {
            // --- 1. Load dataset ARFF ---
            DataSource source = new DataSource("data/heart_improve.arff");
            Instances dataset = source.getDataSet();

            // --- 2. Set class attribute (last column) ---
            if (dataset.classIndex() == -1)
                dataset.setClassIndex(dataset.numAttributes() - 1);

            // --- 3. Initialize Naive Bayes classifier ---
            NaiveBayes nb = new NaiveBayes();

            // --- 4. Evaluate with 10-fold cross-validation ---
            Evaluation eval = new Evaluation(dataset);
            eval.crossValidateModel(nb, dataset, 10, new Random(1));

            // --- 5. Print full WEKA-like results ---
            System.out.println(eval.toSummaryString("\n=== Summary ===\n", true));
            System.out.println(eval.toClassDetailsString("\n=== Detailed Accuracy By Class ===\n"));
            System.out.println(eval.toMatrixString("\n=== Confusion Matrix ===\n"));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
