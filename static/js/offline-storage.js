/**
 * JalScan Offline Storage Module
 * Uses IndexedDB for storing submissions when offline
 * Syncs automatically when internet connection returns
 */

const DB_NAME = 'JalScanOfflineDB';
const DB_VERSION = 1;
const STORE_NAME = 'pendingSubmissions';

let db = null;

/**
 * Initialize IndexedDB database
 */
function openDatabase() {
    return new Promise((resolve, reject) => {
        if (db) {
            resolve(db);
            return;
        }

        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onerror = () => {
            console.error('[OfflineStorage] Failed to open database:', request.error);
            reject(request.error);
        };

        request.onsuccess = () => {
            db = request.result;
            console.log('[OfflineStorage] Database opened successfully');
            resolve(db);
        };

        request.onupgradeneeded = (event) => {
            const database = event.target.result;

            // Create object store for pending submissions
            if (!database.objectStoreNames.contains(STORE_NAME)) {
                const store = database.createObjectStore(STORE_NAME, {
                    keyPath: 'id',
                    autoIncrement: true
                });

                store.createIndex('timestamp', 'timestamp', { unique: false });
                store.createIndex('synced', 'synced', { unique: false });
                console.log('[OfflineStorage] Object store created');
            }
        };
    });
}

/**
 * Save a submission to IndexedDB for later sync
 * @param {Object} submissionData - The form data to save
 * @returns {Promise<number>} - The ID of the saved submission
 */
async function saveSubmission(submissionData) {
    try {
        const database = await openDatabase();

        return new Promise((resolve, reject) => {
            const transaction = database.transaction([STORE_NAME], 'readwrite');
            const store = transaction.objectStore(STORE_NAME);

            const record = {
                ...submissionData,
                timestamp: new Date().toISOString(),
                synced: false,
                attempts: 0
            };

            const request = store.add(record);

            request.onsuccess = () => {
                console.log('[OfflineStorage] Submission saved with ID:', request.result);
                resolve(request.result);
            };

            request.onerror = () => {
                console.error('[OfflineStorage] Failed to save submission:', request.error);
                reject(request.error);
            };
        });
    } catch (error) {
        console.error('[OfflineStorage] Error saving submission:', error);
        throw error;
    }
}

/**
 * Get all pending (unsynced) submissions
 * @returns {Promise<Array>} - Array of pending submissions
 */
async function getPendingSubmissions() {
    try {
        const database = await openDatabase();

        return new Promise((resolve, reject) => {
            const transaction = database.transaction([STORE_NAME], 'readonly');
            const store = transaction.objectStore(STORE_NAME);
            const index = store.index('synced');
            const request = index.getAll(IDBKeyRange.only(false));

            request.onsuccess = () => {
                console.log('[OfflineStorage] Found pending submissions:', request.result.length);
                resolve(request.result);
            };

            request.onerror = () => {
                console.error('[OfflineStorage] Failed to get pending submissions:', request.error);
                reject(request.error);
            };
        });
    } catch (error) {
        console.error('[OfflineStorage] Error getting pending submissions:', error);
        return [];
    }
}

/**
 * Get count of pending submissions
 * @returns {Promise<number>}
 */
async function getPendingCount() {
    try {
        const pending = await getPendingSubmissions();
        return pending.length;
    } catch (error) {
        return 0;
    }
}

/**
 * Mark a submission as synced
 * @param {number} id - The submission ID
 */
async function markAsSynced(id) {
    try {
        const database = await openDatabase();

        return new Promise((resolve, reject) => {
            const transaction = database.transaction([STORE_NAME], 'readwrite');
            const store = transaction.objectStore(STORE_NAME);
            const request = store.get(id);

            request.onsuccess = () => {
                const record = request.result;
                if (record) {
                    record.synced = true;
                    record.syncedAt = new Date().toISOString();
                    store.put(record);
                    console.log('[OfflineStorage] Marked as synced:', id);
                }
                resolve();
            };

            request.onerror = () => {
                reject(request.error);
            };
        });
    } catch (error) {
        console.error('[OfflineStorage] Error marking as synced:', error);
    }
}

/**
 * Delete a synced submission
 * @param {number} id - The submission ID
 */
async function deleteSubmission(id) {
    try {
        const database = await openDatabase();

        return new Promise((resolve, reject) => {
            const transaction = database.transaction([STORE_NAME], 'readwrite');
            const store = transaction.objectStore(STORE_NAME);
            const request = store.delete(id);

            request.onsuccess = () => {
                console.log('[OfflineStorage] Deleted submission:', id);
                resolve();
            };

            request.onerror = () => {
                reject(request.error);
            };
        });
    } catch (error) {
        console.error('[OfflineStorage] Error deleting submission:', error);
    }
}

/**
 * Attempt to sync all pending submissions
 * @returns {Promise<{synced: number, failed: number}>}
 */
async function syncPendingSubmissions() {
    const pending = await getPendingSubmissions();

    if (pending.length === 0) {
        console.log('[OfflineStorage] No pending submissions to sync');
        return { synced: 0, failed: 0 };
    }

    console.log(`[OfflineStorage] Syncing ${pending.length} pending submissions...`);

    let synced = 0;
    let failed = 0;

    for (const submission of pending) {
        try {
            // Create FormData for submission
            const formData = new FormData();
            formData.append('water_level', submission.water_level);
            formData.append('notes', submission.notes || '');
            formData.append('latitude', submission.latitude);
            formData.append('longitude', submission.longitude);
            formData.append('location_verified', submission.location_verified || 'true');
            formData.append('qr_scanned', submission.qr_scanned || 'false');

            // Handle photo - it might be a base64 string
            if (submission.photo) {
                // Convert base64 to Blob
                const response = await fetch(submission.photo);
                const blob = await response.blob();
                formData.append('photo', blob, 'offline_photo.jpg');
            }

            // Submit to server
            const response = await fetch(`/api/submit-reading/${submission.site_id}`, {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                await deleteSubmission(submission.id);
                synced++;
                console.log(`[OfflineStorage] Synced submission ${submission.id}`);
            } else {
                failed++;
                console.error(`[OfflineStorage] Failed to sync ${submission.id}:`, response.status);
            }
        } catch (error) {
            failed++;
            console.error(`[OfflineStorage] Error syncing ${submission.id}:`, error);
        }
    }

    console.log(`[OfflineStorage] Sync complete: ${synced} synced, ${failed} failed`);
    return { synced, failed };
}

/**
 * Check if we're online
 */
function isOnline() {
    return navigator.onLine;
}

/**
 * Register background sync if supported
 */
async function registerBackgroundSync() {
    if ('serviceWorker' in navigator && 'SyncManager' in window) {
        try {
            const registration = await navigator.serviceWorker.ready;
            await registration.sync.register('sync-submissions');
            console.log('[OfflineStorage] Background sync registered');
        } catch (error) {
            console.error('[OfflineStorage] Background sync registration failed:', error);
        }
    }
}

/**
 * Listen for online event and trigger sync
 */
function setupOnlineListener() {
    window.addEventListener('online', async () => {
        console.log('[OfflineStorage] Connection restored, syncing...');

        // Show syncing notification
        if (typeof showAlert === 'function') {
            showAlert('🔄 Connection restored! Syncing offline submissions...', 'info');
        }

        const result = await syncPendingSubmissions();

        if (result.synced > 0) {
            if (typeof showAlert === 'function') {
                showAlert(`✅ Synced ${result.synced} offline submission(s)!`, 'success');
            }

            // Refresh page to show updated data
            setTimeout(() => {
                if (window.location.pathname.includes('dashboard')) {
                    window.location.reload();
                }
            }, 2000);
        }
    });

    window.addEventListener('offline', () => {
        console.log('[OfflineStorage] Connection lost');
        if (typeof showAlert === 'function') {
            showAlert('📴 You are offline. Submissions will be saved locally.', 'warning');
        }
    });
}

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    openDatabase().then(() => {
        console.log('[OfflineStorage] Initialized');
        setupOnlineListener();

        // Try to sync any pending submissions on page load
        if (isOnline()) {
            syncPendingSubmissions();
        }
    });
});

// Export for use in other scripts
window.OfflineStorage = {
    saveSubmission,
    getPendingSubmissions,
    getPendingCount,
    markAsSynced,
    deleteSubmission,
    syncPendingSubmissions,
    isOnline,
    registerBackgroundSync
};
