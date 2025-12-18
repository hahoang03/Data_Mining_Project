package data_preprocessing.data_cleaning;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.CSVSaver;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import java.util.Collections;

public class handle_missingvalue {

    public static void main(String[] args) throws Exception {
        // --- Load dataset ---
        DataSource source = new DataSource("data/heart.csv");
        Instances data = source.getDataSet();

        // Optional: set class index (target)
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // --- Define feature groups ---
        String[] numericFeatures = {"Age", "Blood Pressure", "Cholesterol Level", "BMI",
                                    "Sleep Hours", "Triglyceride Level", "Fasting Blood Sugar",
                                    "CRP Level", "Homocysteine Level"};

        String[] binaryFeatures = {"Gender", "Smoking", "Diabetes", "Family Heart Disease",
                                   "High Blood Pressure", "Low HDL Cholesterol", "High LDL Cholesterol"};

        String[] ordinalFeatures = {"Exercise Habits", "Alcohol Consumption", "Stress Level", "Sugar Consumption"};

        // --- Fill numeric features by mean ---
        for (String colName : numericFeatures) {
            int idx = data.attribute(colName).index();
            double sum = 0;
            int count = 0;

            // Calculate mean
            for (int j = 0; j < data.numInstances(); j++) {
                Instance inst = data.instance(j);
                if (!inst.isMissing(idx)) {
                    sum += inst.value(idx);
                    count++;
                }
            }
            double mean = sum / count;

            // Fill missing
            for (int j = 0; j < data.numInstances(); j++) {
                Instance inst = data.instance(j);
                if (inst.isMissing(idx)) {
                    inst.setValue(idx, mean);
                }
            }
        }

        // --- Fill binary features by mode ---
        for (String colName : binaryFeatures) {
            int idx = data.attribute(colName).index();
            Map<String, Integer> freq = new HashMap<>();

            // Count frequency
            for (int j = 0; j < data.numInstances(); j++) {
                Instance inst = data.instance(j);
                if (!inst.isMissing(idx)) {
                    String val = inst.stringValue(idx);
                    freq.put(val, freq.getOrDefault(val, 0) + 1);
                }
            }

            // Find mode
            String mode = null;
            int maxCount = -1;
            for (Map.Entry<String, Integer> entry : freq.entrySet()) {
                if (entry.getValue() > maxCount) {
                    maxCount = entry.getValue();
                    mode = entry.getKey();
                }
            }

            // Fill missing with mode
            for (int j = 0; j < data.numInstances(); j++) {
                Instance inst = data.instance(j);
                if (inst.isMissing(idx)) {
                    inst.setValue(idx, mode);
                }
            }
        }

        // --- Fill ordinal features by median (numeric mapping) ---
        Map<String, Map<String, Integer>> ordinalMapping = new HashMap<>();
        ordinalMapping.put("Exercise Habits", Map.of("Low",0,"Medium",1,"High",2));
        ordinalMapping.put("Alcohol Consumption", Map.of("None",0,"Low",1,"Medium",2,"High",3));
        ordinalMapping.put("Stress Level", Map.of("Low",0,"Medium",1,"High",2));
        ordinalMapping.put("Sugar Consumption", Map.of("Low",0,"Medium",1,"High",2));

        for (String colName : ordinalFeatures) {
            int idx = data.attribute(colName).index();
            ArrayList<Integer> values = new ArrayList<>();

            // Convert existing values to numeric
            for (int j = 0; j < data.numInstances(); j++) {
                Instance inst = data.instance(j);
                if (!inst.isMissing(idx)) {
                    values.add(ordinalMapping.get(colName).get(inst.stringValue(idx)));
                }
            }

            // Calculate median
            Collections.sort(values);
            int medianVal;
            int n = values.size();
            if (n % 2 == 0) {
                medianVal = (values.get(n/2 -1) + values.get(n/2)) / 2;
            } else {
                medianVal = values.get(n/2);
            }

            // Fill missing with median (numeric)
            for (int j = 0; j < data.numInstances(); j++) {
                Instance inst = data.instance(j);
                if (inst.isMissing(idx)) {
                    inst.setValue(idx, medianVal);
                }
            }
        }

        // --- Save filled dataset to CSV ---
        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);
        saver.setFile(new File("data/heart_filled.csv"));
        saver.writeBatch();

        System.out.println("Missing values handled and saved to heart_filled.csv");
    }
}
