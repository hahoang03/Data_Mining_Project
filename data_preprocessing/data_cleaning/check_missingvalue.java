package data_preprocessing.data_cleaning;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class check_missingvalue {
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("data/heart_filled.csv");
        Instances data = source.getDataSet();

        // Nếu chưa set class index
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // In header bảng
        System.out.printf("%-30s | %s%n", "Attribute", "Missing Values");
        System.out.println("-----------------------------------------------");

        // Duyệt từng attribute
        for (int i = 0; i < data.numAttributes(); i++) {
            int missingCount = 0;
            for (int j = 0; j < data.numInstances(); j++) {
                Instance inst = data.instance(j);
                if (inst.isMissing(i)) {
                    missingCount++;
                }
            }
            System.out.printf("%-30s | %d%n", data.attribute(i).name(), missingCount);
        }
    }
}
