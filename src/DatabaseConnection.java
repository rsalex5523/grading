import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public class DatabaseConnection {

    // --- database connection details ---
    private static final String DB_URL = "jdbc:mysql://rs-vm06:3306/20250722recordsale";
    private static final String USER = "pricing";
    private static final String PASSWORD = "SuperCheap199";

    public static void main(String[] args) {
        // The SQL query you want to execute
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
//                "WHERE ri.recordID = ? ;";

        // Using try-with-resources to ensure resources are closed automatically
        try (Connection conn = DriverManager.getConnection(DB_URL, USER, PASSWORD);
             PreparedStatement pstmt = conn.prepareStatement(sqlQuery)) {

            // Set parameters for the query (this prevents SQL injection)
            // Here, we are looking for users in "Berlin"
            //stmt.setString(1, "507050028");

            // Execute the query and get the results
            ResultSet rs = pstmt.executeQuery();

            System.out.println("--- User Data from Database ---");
            // Loop through the results row by row

            while (rs.next()) {
                // Retrieve data from each column by its name or index
                int record_ID = rs.getInt("recordID");
                float BP = rs.getFloat("BP");
                float FinalPrice = rs.getFloat("FinalPrice");
                String coverGrading = rs.getString("coverGrading");
                String mediaGrading = rs.getString("mediaGrading");
                int mediaDiscount = rs.getInt("mediaDiscount");
                int coverDiscount = rs.getInt("coverDiscount");

                // Process the retrieved data (e.g., print it)
                System.out.printf("recordID: %d, BP: %.2f, FinalPrice: %.2f, coverGrading: %s, mediaGrading: %s, mediaDiscount: %d, coverDiscount: %d%n", record_ID, BP, FinalPrice, coverGrading, mediaGrading, mediaDiscount, coverDiscount);
            }
            System.out.println("-----------------------------");


        } catch (SQLException e) {
            System.err.println("Database connection or query failed!");
            e.printStackTrace();
        }
    }
}