// DynamicGrading.java
// Alexandra Johansen
// Created on 24.07.2025
import java.lang.Math;
public class DynamicGrading {

    public enum GradingType {
        MEDIA,
        COVER
    }

    public int calculateGradingAdjustment(double basePrice, String grading, GradingType type) {
        int priceGroup = getPriceGroup(basePrice);
        double scalingFactor = getScalingFactor(grading, type);
        double adjustmentFactor = 0.0; // to be casted to int!

        // Cap the price group at 5 to treat Q6 the same as Q5
        int effectivePriceGroup = Math.min(priceGroup, 5);

        if (type == GradingType.MEDIA) {
            // Apply the specific formula based on the Media grading
            switch (grading) {
                case "NM": // Uses Cosine formula with addition
                    adjustmentFactor = scalingFactor * Math.cos(((double) effectivePriceGroup - 1) / 4.0 * Math.PI);
                    break;
                case "EX": // Uses Parabolic formula with subtraction
                    adjustmentFactor = -scalingFactor * Math.pow(effectivePriceGroup - 3.5, 2);
                    break;
                case "VG": // Uses Parabolic formula with subtraction
                    adjustmentFactor = -scalingFactor * Math.pow(effectivePriceGroup - 3.5, 2);
                    break;
                // Uses Cosine formula with subtraction - 'F' and 'P' use the same logic as 'G'
                case "F":
                case "P":
                case "G":
                    adjustmentFactor = -scalingFactor * Math.cos(((double) effectivePriceGroup - 1) / 4.0 * Math.PI);
                    break;
            }
        } else { // type is COVER
            // Cover always uses the Linear formula with subtraction
            adjustmentFactor = -scalingFactor * (2.25 * (effectivePriceGroup - 1));
        }
        return (int) Math.round(adjustmentFactor);
    } // calculateGradingAdjustment()

    /* Price Groups */
    private int getPriceGroup(double basePrice) {
        if (basePrice >= 0 && basePrice < 10) return 1;
        if (basePrice >= 10 && basePrice < 16) return 2;
        if (basePrice >= 16 && basePrice < 23) return 3;
        if (basePrice >= 23 && basePrice < 37) return 4;
        if (basePrice >= 37 && basePrice < 200) return 5;
        if (basePrice >= 200) return 6;
        return 0;
    }

    /* Scaling factor */
    private double getScalingFactor(String grading, GradingType type) {
        if (type == GradingType.MEDIA) {
            switch (grading) {
                case "NM":
                    return 10.0;
                case "EX":
                    return 1.0;
                case "VG":
                    return 1.0;
                // --- NEW: 'F' and 'P' get the same scaling factor as 'G' ---
                case "F":
                case "P":
                case "G":
                    return 7.0;
                default:
                    return 0.0;
            }
        } else { // type is COVER
            switch (grading) {
                case "NM":
                    return 0.5;
                case "EX":
                    return 0.5;
                case "VG":
                    return 0.5;
                // --- NEW: 'F' and 'P' get the same scaling factor as 'G' ---
                case "F":
                case "P":
                case "G":
                    return 0.3;
                default:
                    return 0.0;
            }
        }
    }


}