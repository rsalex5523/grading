import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public class Test {
    // --- Replace with your actual database connection details ---
    private static final String DB_URL = "jdbc:mysql://rs-vm06:3306/20250722recordsale";
    private static final String USER = "pricing";
    private static final String PASSWORD = "SuperCheap199";

    public static void main(String[] args) {
        // ... (SQL query is the same) ...
        String sqlQuery = "SELECT " + // REMEMBER SPACE BEFORE "!
                "ri.recordID, " +
                "ri.mediaGrading, " +
                "ri.covergrading, " +
                "rr.preAuctionPrice + rr.addPriceByAge AS BP, " +
                "rt.FinalPrice, " +
                "rgm.reducePriceMediaLP AS mediaDiscount, " +
                "rgc.reducePriceCoverLP AS coverDiscount " +

                "FROM rsitems as ri " +

                "JOIN rsrecords AS rr ON ri.recordID = rr.ID " +
                "JOIN rstransactions AS rt on rt.ItemID = ri.ID " +
                "JOIN rsgrading AS rgm ON ri.mediaGrading = rgm.grading " +
                "JOIN rsgrading AS rgc ON ri.coverGrading = rgc.grading " +

                "WHERE rr.mainMedia = 'Vinyl' " +
                "AND ri.mediaGrading IN ('NM', 'EX', 'VG', 'G', 'P', 'F')" +
                "AND ri.coverGrading IN ('NM', 'EX', 'VG', 'G', 'P', 'F')" +
                "LIMIT 100; " ;

        DynamicGrading priceCalculator = new DynamicGrading();

        try (Connection conn = DriverManager.getConnection(DB_URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sqlQuery)) {

            ResultSet rs = pstmt.executeQuery();
            System.out.println("--- Calculated Prices (Using Hybrid Formula Model) ---");

            while (rs.next()) {
                float BP = rs.getFloat("BP");
                String mediaGrading = rs.getString("mediaGrading");
                String coverGrading = rs.getString("coverGrading");
                int mediaDiscount_base = rs.getInt("mediaDiscount");
                int coverDiscount_base = rs.getInt("coverDiscount");

                // These variables are now ints
                int mediaAdjustment = priceCalculator.calculateGradingAdjustment(BP, mediaGrading, DynamicGrading.GradingType.MEDIA);
                int coverAdjustment = priceCalculator.calculateGradingAdjustment(BP, coverGrading, DynamicGrading.GradingType.COVER);

                // The rest of the calculation proceeds with the integer adjustments
                double newMediaDiscount = mediaDiscount_base + mediaAdjustment;
                double newCoverDiscount = coverDiscount_base + coverAdjustment;
                double newFinalPrice = BP + newMediaDiscount + newCoverDiscount;

                // Print results
                System.out.printf("Record ID: %d, Base Price: %.2f%n", rs.getInt("recordID"), BP);
                System.out.printf("  > Media: Grade=%s, Base Discount=%d, Adjustment=%d, New Discount=%.2f%n", mediaGrading, mediaDiscount_base, mediaAdjustment, newMediaDiscount);
                System.out.printf("  > Cover: Grade=%s, Base Discount=%d, Adjustment=%d, New Discount=%.2f%n", coverGrading, coverDiscount_base, coverAdjustment, newCoverDiscount);
                System.out.printf("  > New Final Price: %.2f%n", newFinalPrice);
                System.out.println("----------------------------------------------------------");
            }

        } catch (SQLException e) {
            System.err.println("Database connection or query failed!");
            e.printStackTrace();
        }
    }
}