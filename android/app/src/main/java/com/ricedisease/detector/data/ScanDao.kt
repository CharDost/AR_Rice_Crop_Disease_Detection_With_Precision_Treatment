package com.ricedisease.detector.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

/**
 * Room DAO for ScanRecord CRUD operations.
 */
@Dao
interface ScanDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(record: ScanRecord): Long

    @Delete
    suspend fun delete(record: ScanRecord)

    @Query("DELETE FROM scan_records")
    suspend fun deleteAll()

    @Query("SELECT * FROM scan_records ORDER BY timestamp DESC")
    fun getAllScans(): Flow<List<ScanRecord>>

    @Query("SELECT * FROM scan_records WHERE isHealthy = 1 ORDER BY timestamp DESC")
    fun getHealthyScans(): Flow<List<ScanRecord>>

    @Query("SELECT * FROM scan_records WHERE isHealthy = 0 AND isValidInput = 1 ORDER BY timestamp DESC")
    fun getDiseasedScans(): Flow<List<ScanRecord>>

    @Query("""
        SELECT * FROM scan_records
        WHERE timestamp BETWEEN :startMs AND :endMs
        ORDER BY timestamp DESC
    """)
    fun getScansByDateRange(startMs: Long, endMs: Long): Flow<List<ScanRecord>>

    @Query("""
        SELECT * FROM scan_records
        WHERE predictedClass LIKE :diseaseFilter
        ORDER BY timestamp DESC
    """)
    fun getScansByDisease(diseaseFilter: String): Flow<List<ScanRecord>>

    @Query("SELECT COUNT(*) FROM scan_records")
    suspend fun getTotalCount(): Int

    @Query("SELECT COUNT(*) FROM scan_records WHERE isHealthy = 1")
    suspend fun getHealthyCount(): Int

    @Query("SELECT COUNT(*) FROM scan_records WHERE isHealthy = 0 AND isValidInput = 1")
    suspend fun getDiseasedCount(): Int

    /** Last N scans for trend chart */
    @Query("SELECT * FROM scan_records WHERE isValidInput = 1 ORDER BY timestamp DESC LIMIT :n")
    suspend fun getRecentScans(n: Int = 30): List<ScanRecord>
}
