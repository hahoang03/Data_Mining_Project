import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;

import java.util.Random;

public class ClassificationExample {

    public static void main(String[] args) throws Exception {
        // 1. Load dataset từ file ARFF
        DataSource source = new DataSource("data/dataset.arff");
        Instances dataset = source.getDataSet();

        // 2. Nếu dataset chưa có class index (label), đặt class index là cột cuối
        if (dataset.classIndex() == -1)
            dataset.setClassIndex(dataset.numAttributes() - 1);

        // 3. Chọn thuật toán: Decision Tree (J48)
        // Decision Tree dễ hiểu, giải thích kết quả trực quan, phù hợp với dữ liệu nhỏ đến trung bình
        Classifier model = new J48(); // tạo model J48

        // 4. Huấn luyện model
        model.buildClassifier(dataset);

        // 5. Đánh giá model bằng 10-fold cross-validation
        Evaluation eval = new Evaluation(dataset);
        eval.crossValidateModel(model, dataset, 10, new Random(1));

        // 6. In kết quả
        System.out.println("=== Summary ===");
        System.out.println(eval.toSummaryString());
        System.out.println("=== Confusion Matrix ===");
        double[][] cm = eval.confusionMatrix();
        for (int i = 0; i < cm.length; i++) {
            for (int j = 0; j < cm[i].length; j++) {
                System.out.print(cm[i][j] + " ");
            }
            System.out.println();
        }

        // 7. In cây quyết định
        System.out.println("=== Decision Tree ===");
        System.out.println(model);
    }
}
