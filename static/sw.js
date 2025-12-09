const CACHE_NAME = 'jalscan-v13';
const STATIC_CACHE = 'jalscan-static-v13';
const DYNAMIC_CACHE = 'jalscan-dynamic-v13';

// Static assets to cache immediately
const STATIC_ASSETS = [
  '/',
  '/login',
  '/offline',
  '/dashboard',
  '/static/css/style.css',
  '/static/js/i18n.js',
  '/static/js/offline-storage.js',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
  '/static/manifest.json',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
  'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('[Service Worker] Installing...');
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[Service Worker] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => {
        console.log('[Service Worker] Static assets cached');
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('[Service Worker] Cache failed:', error);
      })
  );
});

// Activate event - clean old caches
self.addEventListener('activate', (event) => {
  console.log('[Service Worker] Activating...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== STATIC_CACHE && name !== DYNAMIC_CACHE)
          .map((name) => {
            console.log('[Service Worker] Deleting old cache:', name);
            return caches.delete(name);
          })
      );
    }).then(() => {
      console.log('[Service Worker] Activated');
      return self.clients.claim();
    })
  );
});

// Fetch event - network first, fallback to cache
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // Skip API calls and form submissions
  if (url.pathname.startsWith('/api/') || url.pathname.includes('/submit')) {
    return;
  }

  // For static assets, use cache-first strategy
  if (request.url.includes('/static/') || request.url.includes('cdn.jsdelivr.net')) {
    event.respondWith(
      caches.match(request).then((cachedResponse) => {
        if (cachedResponse) {
          return cachedResponse;
        }
        return fetch(request).then((response) => {
          if (response.ok) {
            const responseClone = response.clone();
            caches.open(STATIC_CACHE).then((cache) => {
              cache.put(request, responseClone);
            });
          }
          return response;
        });
      })
    );
    return;
  }

  // For HTML pages, use network-first strategy with cache fallback
  event.respondWith(
    fetch(request)
      .then((response) => {
        if (response.ok) {
          const responseClone = response.clone();
          caches.open(DYNAMIC_CACHE).then((cache) => {
            cache.put(request, responseClone);
          });
        }
        return response;
      })
      .catch(() => {
        return caches.match(request).then((cachedResponse) => {
          if (cachedResponse) {
            return cachedResponse;
          }
          // Return offline page for HTML requests
          if (request.headers.get('accept') && request.headers.get('accept').includes('text/html')) {
            return caches.match('/offline');
          }
        });
      })
  );
});

// Handle background sync for offline submissions
self.addEventListener('sync', (event) => {
  console.log('[Service Worker] Sync event:', event.tag);
  if (event.tag === 'sync-submissions') {
    event.waitUntil(syncSubmissions());
  }
});

// Handle push notifications
self.addEventListener('push', (event) => {
  console.log('[Service Worker] Push received');
  const options = {
    body: event.data ? event.data.text() : 'New notification from JalScan',
    icon: '/static/icons/icon-192.png',
    badge: '/static/icons/icon-192.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    }
  };

  event.waitUntil(
    self.registration.showNotification('JalScan Alert', options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  console.log('[Service Worker] Notification clicked');
  event.notification.close();
  event.waitUntil(
    clients.openWindow('/')
  );
});

// Sync submissions that were made offline
async function syncSubmissions() {
  try {
    console.log('[Service Worker] Syncing offline submissions...');

    // Open IndexedDB
    const db = await openIndexedDB();
    const pending = await getPendingFromDB(db);

    if (pending.length === 0) {
      console.log('[Service Worker] No pending submissions');
      return;
    }

    console.log(`[Service Worker] Found ${pending.length} pending submissions`);

    for (const submission of pending) {
      try {
        // Create FormData
        const formData = new FormData();
        formData.append('water_level', submission.water_level);
        formData.append('notes', submission.notes || '');
        formData.append('latitude', submission.latitude);
        formData.append('longitude', submission.longitude);
        formData.append('location_verified', submission.location_verified || 'true');
        formData.append('qr_scanned', submission.qr_scanned || 'false');

        // Handle photo if present
        if (submission.photo) {
          const response = await fetch(submission.photo);
          const blob = await response.blob();
          formData.append('photo', blob, 'offline_photo.jpg');
        }

        // Submit
        const response = await fetch(`/api/submit-reading/${submission.site_id}`, {
          method: 'POST',
          body: formData
        });

        if (response.ok) {
          await deleteFromDB(db, submission.id);
          console.log(`[Service Worker] Synced submission ${submission.id}`);
        }
      } catch (error) {
        console.error(`[Service Worker] Failed to sync ${submission.id}:`, error);
      }
    }

    // Notify clients
    const clients = await self.clients.matchAll();
    clients.forEach(client => {
      client.postMessage({ type: 'SYNC_COMPLETE' });
    });

  } catch (error) {
    console.error('[Service Worker] Sync failed:', error);
  }
}

// IndexedDB helpers for service worker
function openIndexedDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('JalScanOfflineDB', 1);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
  });
}

function getPendingFromDB(db) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['pendingSubmissions'], 'readonly');
    const store = transaction.objectStore('pendingSubmissions');
    const index = store.index('synced');
    const request = index.getAll(IDBKeyRange.only(false));
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result || []);
  });
}

function deleteFromDB(db, id) {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['pendingSubmissions'], 'readwrite');
    const store = transaction.objectStore('pendingSubmissions');
    const request = store.delete(id);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve();
  });
}