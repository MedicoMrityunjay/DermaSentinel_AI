
// --- 1. VISUAL FX ENGINE (Three.js + Anime.js) ---

// A. Three.js Background (Neural Network / Particles)
const initThreeJS = () => {
    if (window.innerWidth < 768) return; // Disable on mobile

    const canvas = document.getElementById('bg-canvas');
    const scene = new THREE.Scene();
    
    // Fog for depth (Slate-900 match)
    scene.fog = new THREE.FogExp2(0x0f172a, 0.002);

    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 50;

    const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Particles
    const geometry = new THREE.BufferGeometry();
    const particlesCount = 700;
    const posArray = new Float32Array(particlesCount * 3);

    for(let i = 0; i < particlesCount * 3; i++) {
        posArray[i] = (Math.random() - 0.5) * 100; // Spread
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));

    const material = new THREE.PointsMaterial({
        size: 0.15,
        color: 0x3b82f6, // Blue-500
        transparent: true,
        opacity: 0.8,
    });

    const particlesMesh = new THREE.Points(geometry, material);
    scene.add(particlesMesh);

    // Mouse Interaction
    let mouseX = 0;
    let mouseY = 0;

    document.addEventListener('mousemove', (event) => {
        mouseX = event.clientX / window.innerWidth - 0.5;
        mouseY = event.clientY / window.innerHeight - 0.5;
    });

    // Animation Loop
    const animate = () => {
        requestAnimationFrame(animate);

        // Rotation
        particlesMesh.rotation.y += 0.001;
        particlesMesh.rotation.x += 0.0005;

        // Mouse Parallax
        particlesMesh.rotation.y += mouseX * 0.05;
        particlesMesh.rotation.x += mouseY * 0.05;

        renderer.render(scene, camera);
    };

    animate();

    // Resize Handler
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
};

// B. Anime.js Entry Animations
const runEntryAnimations = () => {
    anime({
        targets: '#sidebar',
        translateX: [-250, 0],
        opacity: [0, 1],
        easing: 'easeOutExpo',
        duration: 1000,
        delay: 200
    });

    anime({
        targets: '.anime-entry',
        translateY: [20, 0],
        opacity: [0, 1],
        easing: 'easeOutExpo',
        duration: 800,
        delay: anime.stagger(150, {start: 500}) // Stagger effect
    });
};

// --- 2. APP LOGIC (Original Functionality Preserved) ---

const ctx = document.getElementById('abcdChart').getContext('2d');
Chart.defaults.font.family = 'Inter';
Chart.defaults.color = '#64748b';

// Dark Mode Check
if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.classList.add('dark');
} else {
    document.documentElement.classList.remove('dark');
}

const abcdChart = new Chart(ctx, {
    type: 'radar',
    data: {
        labels: ['Asymmetry', 'Border', 'Color'],
        datasets: [{
            label: 'Score',
            data: [0, 0, 0],
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderColor: '#3b82f6',
            borderWidth: 2,
            pointBackgroundColor: '#fff',
            pointBorderColor: '#3b82f6',
            pointRadius: 3
        }]
    },
    options: {
        scales: {
            r: {
                angleLines: { display: false },
                grid: { color: document.documentElement.classList.contains('dark') ? '#334155' : '#f1f5f9' },
                pointLabels: { font: { size: 10, weight: '600' }, color: '#94a3b8' },
                suggestedMin: 0,
                suggestedMax: 1,
                ticks: { display: false }
            }
        },
        plugins: { legend: { display: false } },
        maintainAspectRatio: false
    }
});

// Theme Toggle Logic
function toggleTheme() {
    if (document.documentElement.classList.contains('dark')) {
        document.documentElement.classList.remove('dark');
        localStorage.theme = 'light';
        abcdChart.options.scales.r.grid.color = '#f1f5f9';
    } else {
        document.documentElement.classList.add('dark');
        localStorage.theme = 'dark';
        abcdChart.options.scales.r.grid.color = '#334155';
    }
    abcdChart.update();
}

document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
document.getElementById('theme-toggle-mobile').addEventListener('click', toggleTheme);

// --- App Logic ---
const fileInput = document.getElementById('file-upload');
const imageContainer = document.getElementById('image-preview-container');
const imagePreview = document.getElementById('image-preview');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const uploadCard = document.getElementById('upload-card');
const maskOverlay = document.getElementById('mask-overlay');
const scanOverlay = document.getElementById('scan-overlay');
const analyzeBtn = document.getElementById('analyze-btn');
const downloadBtn = document.getElementById('download-btn');
const resetBtn = document.getElementById('reset-btn');
const toggleMaskCheckbox = document.getElementById('toggle-mask');
const statusBadge = document.getElementById('status-badge');
const caseIdEl = document.getElementById('case-id');
const errorToast = document.getElementById('error-toast');
const errorMsg = document.getElementById('error-msg');
const ageInput = document.getElementById('age');

let currentScanId = null;

// Validation Helper
function showError(msg) {
    errorMsg.innerText = msg;
    errorToast.classList.remove('hidden');
    setTimeout(() => errorToast.classList.add('hidden'), 4000);
}

// Age Input Constraint
if (ageInput) {
    ageInput.addEventListener('input', function() {
        if (this.value.length > 3) this.value = this.value.slice(0, 3);
    });
}

function validateField(id, condition = null) {
    const el = document.getElementById(id);
    let isValid = true;

    if (!el.value) {
        isValid = false;
    } else if (condition && !condition(el.value)) {
        isValid = false;
    }

    if (!isValid) {
        el.classList.add('border-red-500', 'ring-2', 'ring-red-200', 'animate-shake');
        setTimeout(() => el.classList.remove('animate-shake'), 500);
        return false;
    } else {
        el.classList.remove('border-red-500', 'ring-2', 'ring-red-200');
        return true;
    }
}

// Upload
fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadPlaceholder.classList.add('hidden');
            imageContainer.classList.remove('hidden');
            maskOverlay.style.opacity = 0;
            if (toggleMaskCheckbox) toggleMaskCheckbox.checked = false;
            statusBadge.innerText = "Image Loaded";
            statusBadge.className = "px-3 py-1 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs font-semibold border border-blue-200 dark:border-blue-800";
            uploadCard.classList.remove('border-red-500', 'ring-2', 'ring-red-200');
        }
        reader.readAsDataURL(e.target.files[0]);
    }
});

// Analyze
analyzeBtn.addEventListener('click', async () => {
    // 1. Strict Validation
    let isValid = true;

    // Check Age (0-120)
    if (!validateField('age', val => val >= 0 && val <= 120)) {
        isValid = false;
        if (document.getElementById('age').value && (document.getElementById('age').value < 0 || document.getElementById('age').value > 120)) {
            showError("Invalid Age (0-120).");
            return;
        }
    }

    if (!validateField('sex')) isValid = false;
    if (!validateField('site')) isValid = false;

    if (!fileInput.files[0]) {
        uploadCard.classList.add('border-red-500', 'ring-2', 'ring-red-200', 'animate-shake');
        setTimeout(() => uploadCard.classList.remove('animate-shake'), 500);
        isValid = false;
    } else {
        uploadCard.classList.remove('border-red-500', 'ring-2', 'ring-red-200');
    }

    if (!isValid) {
        showError("Please complete all required fields.");
        return;
    }

    // 2. Processing with Safety Net
    const originalText = analyzeBtn.innerHTML;

    try {
        analyzeBtn.innerHTML = `<span>Analyzing...</span>`;
        analyzeBtn.disabled = true;
        if (resetBtn) resetBtn.disabled = true;
        scanOverlay.classList.remove('hidden');

        statusBadge.innerText = "Processing...";
        statusBadge.className = "px-3 py-1 rounded-full bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 text-xs font-semibold border border-amber-200 dark:border-amber-800 animate-pulse";

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('age', document.getElementById('age').value);
        formData.append('sex', document.getElementById('sex').value);
        formData.append('site', document.getElementById('site').value);

        // Clear inputs
        document.getElementById('age').value = '';
        document.getElementById('age').classList.remove('border-red-500', 'ring-2', 'ring-red-200');

        document.getElementById('sex').selectedIndex = 0;
        document.getElementById('sex').classList.remove('border-red-500', 'ring-2', 'ring-red-200');

        document.getElementById('site').selectedIndex = 0;
        document.getElementById('site').classList.remove('border-red-500', 'ring-2', 'ring-red-200');

        caseIdEl.innerText = "PENDING";
        statusBadge.innerText = "Ready";
        statusBadge.className = "px-3 py-1 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-300 text-xs font-semibold border border-slate-200 dark:border-slate-600";

        document.getElementById('diagnosis').innerText = "--";
        document.getElementById('diagnosis').className = "text-3xl font-bold text-slate-900 dark:text-white";

        document.getElementById('probability').innerText = "0.0%";
        document.getElementById('prob-bar').style.width = "0%";

        const skinEl = document.getElementById('skin-type-display');
        if (skinEl) { skinEl.innerText = "--"; skinEl.classList.remove('text-blue-600', 'dark:text-blue-400'); }

        document.getElementById('clinical-notes').innerText = "Waiting for analysis...";
        abcdChart.data.datasets[0].data = [0, 0, 0];
        abcdChart.update();

        if (downloadBtn) downloadBtn.disabled = true;
        currentScanId = null;

        // Call API (No Auth)
        const res = await fetch('/diagnose', {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || "Analysis request failed");
        }

        const data = await res.json();

        if (data.status === 'REJECTED') {
            showError(data.reason);
            statusBadge.innerText = "Rejected";
            statusBadge.className = "px-3 py-1 rounded-full bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 text-xs font-semibold border border-red-200 dark:border-red-800";
            return;
        }

        statusBadge.innerText = "Analysis Complete";
        statusBadge.className = "px-3 py-1 rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 text-xs font-semibold border border-green-200 dark:border-green-800";
        scanOverlay.classList.add('hidden');

        currentScanId = data.scan_id;
        fetchHistory();

        // Update UI
        caseIdEl.innerText = currentScanId;
        const isMelanoma = data.diagnosis === 'Melanoma';
        document.getElementById('diagnosis').innerText = data.diagnosis;
        document.getElementById('diagnosis').className = isMelanoma ?
            "text-3xl font-bold text-red-600 dark:text-red-400" :
            "text-3xl font-bold text-green-600 dark:text-green-400";

        const probPct = (data.probability * 100).toFixed(1) + "%";
        document.getElementById('probability').innerText = probPct;
        document.getElementById('prob-bar').style.width = probPct;

        if (skinEl) {
            skinEl.innerText = data.skin_type || "Unknown";
            skinEl.classList.add('text-blue-600', 'dark:text-blue-400');
        }
        
        // Bias Warning
        const biasWarning = document.getElementById('bias-warning');
        if (data.bias_warning) {
            biasWarning.classList.remove('hidden');
        } else {
            biasWarning.classList.add('hidden');
        }
        
        // Uncertainty
        const uncEl = document.getElementById('uncertainty-text');
        if (uncEl) uncEl.innerText = (data.uncertainty_score * 100).toFixed(1) + "%";

        document.getElementById('clinical-notes').innerText = data.clinical_note || "No notes available.";

        if (data.concepts) {
            abcdChart.data.datasets[0].data = [
                data.concepts.asymmetry || 0,
                data.concepts.border || 0,
                data.concepts.color || 0
            ];
            abcdChart.update();
        }
        
        // Mask
        if (data.mask_base64) {
            maskOverlay.src = `data:image/png;base64,${data.mask_base64}`;
        }

        if (downloadBtn) downloadBtn.disabled = false;

    } catch (e) {
        showError(e.message || "Server Offline or Connection Failed.");
        statusBadge.innerText = "Connection Error";
        statusBadge.className = "px-3 py-1 rounded-full bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 text-xs font-semibold border border-red-200 dark:border-red-800";
    } finally {
        analyzeBtn.innerHTML = originalText;
        analyzeBtn.disabled = false;
        if (resetBtn) resetBtn.disabled = false;
        if (!currentScanId) scanOverlay.classList.add('hidden');
    }
});

// Toggle Mask
function toggleMask(checked) {
    maskOverlay.style.opacity = checked ? 1 : 0;
    if (toggleMaskCheckbox) toggleMaskCheckbox.checked = checked;
}

if (toggleMaskCheckbox) toggleMaskCheckbox.addEventListener('change', (e) => toggleMask(e.target.checked));

// Download
if (downloadBtn) downloadBtn.addEventListener('click', () => {
    if (currentScanId) window.open(`/report/${currentScanId}`, '_blank');
});

// History Logic (API)
async function fetchHistory() {
    const historyList = document.getElementById('history-list');
    if (!historyList) return;

    try {
        const res = await fetch('/history');
        const scans = await res.json();

        historyList.innerHTML = '';
        if (scans.length === 0) {
            historyList.innerHTML = '<div class="text-xs text-slate-400 text-center py-4">No recent scans</div>';
            return;
        }

        scans.forEach(scan => {
            const item = document.createElement('div');
            item.className = "p-3 bg-white dark:bg-slate-800 rounded-lg border border-slate-100 dark:border-slate-700 hover:border-brand-200 dark:hover:border-brand-800 cursor-pointer transition-colors group";

            const dotColor = scan.diagnosis === 'Melanoma' ? 'bg-red-500' : 'bg-green-500';

            item.innerHTML = `
                <div class="flex justify-between items-start mb-1">
                    <div class="flex items-center gap-2">
                        <span class="w-2 h-2 rounded-full ${dotColor}"></span>
                        <span class="font-semibold text-slate-700 dark:text-slate-200 text-sm">${scan.diagnosis}</span>
                    </div>
                    <span class="text-[10px] text-slate-400">${scan.date}</span>
                </div>
                <div class="text-xs text-slate-500 dark:text-slate-400 pl-4">
                    ID: <span class="font-mono">${scan.id.substring(0, 8)}...</span>
                </div>
            `;

            item.addEventListener('click', () => {
                window.open(`/report/${scan.id}`, '_blank');
            });

            historyList.appendChild(item);
        });
    } catch (e) {
        // Silent fail for history
    }
}

// Reset UI Function
function resetUI() {
    fileInput.value = ''; // Clear selected file
    uploadPlaceholder.classList.remove('hidden');
    imageContainer.classList.add('hidden');
    imagePreview.src = '';
    maskOverlay.style.opacity = 0;
    if (toggleMaskCheckbox) toggleMaskCheckbox.checked = false;
    scanOverlay.classList.add('hidden');

    document.getElementById('age').value = '';
    document.getElementById('age').classList.remove('border-red-500', 'ring-2', 'ring-red-200');

    document.getElementById('sex').selectedIndex = 0;
    document.getElementById('sex').classList.remove('border-red-500', 'ring-2', 'ring-red-200');

    document.getElementById('site').selectedIndex = 0;
    document.getElementById('site').classList.remove('border-red-500', 'ring-2', 'ring-red-200');

    caseIdEl.innerText = "N/A";
    statusBadge.innerText = "Ready";
    statusBadge.className = "px-3 py-1 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-300 text-xs font-semibold border border-slate-200 dark:border-slate-600";

    document.getElementById('diagnosis').innerText = "--";
    document.getElementById('diagnosis').className = "text-3xl font-bold text-slate-900 dark:text-white";

    document.getElementById('probability').innerText = "0.0%";
    document.getElementById('prob-bar').style.width = "0%";

    const skinEl = document.getElementById('skin-type-display');
    if (skinEl) { skinEl.innerText = "--"; skinEl.classList.remove('text-blue-600', 'dark:text-blue-400'); }
    
    document.getElementById('bias-warning').classList.add('hidden');
    document.getElementById('uncertainty-text').innerText = "--";

    document.getElementById('clinical-notes').innerText = "Waiting for analysis...";
    abcdChart.data.datasets[0].data = [0, 0, 0];
    abcdChart.update();

    if (downloadBtn) downloadBtn.disabled = true;
    currentScanId = null;
    analyzeBtn.disabled = false;
    if (resetBtn) resetBtn.disabled = false;
}

if (resetBtn) resetBtn.addEventListener('click', resetUI);

// Chat
const toggleChat = document.getElementById('toggle-chat');
const chatWindow = document.getElementById('chat-window');
const closeChat = document.getElementById('close-chat');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatHistory = document.getElementById('chat-history');

toggleChat.addEventListener('click', () => chatWindow.classList.toggle('hidden'));
closeChat.addEventListener('click', () => chatWindow.classList.add('hidden'));

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const q = chatInput.value.trim();
    if (!q) return;

    chatHistory.innerHTML += `<div class="bg-brand-50 dark:bg-brand-900/20 p-2 rounded-lg rounded-tr-none self-end text-right text-slate-700 dark:text-slate-200 max-w-[90%]"><span class="block text-[10px] font-bold text-brand-600 dark:text-brand-400 uppercase">You</span>${q}</div>`;
    chatInput.value = '';
    chatHistory.scrollTop = chatHistory.scrollHeight;

    if (!fileInput.files[0]) {
        chatHistory.innerHTML += `<div class="bg-slate-50 dark:bg-slate-700 p-2 rounded-lg rounded-tl-none self-start text-slate-600 dark:text-slate-300 max-w-[90%]"><span class="block text-[10px] font-bold text-slate-400 uppercase">AI</span>Please load an image first.</div>`;
        return;
    }

    try {
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('question', q);

        const res = await fetch('/ask', { method: 'POST', body: formData });
        const data = await res.json();
        chatHistory.innerHTML += `<div class="bg-slate-50 dark:bg-slate-700 p-2 rounded-lg rounded-tl-none self-start text-slate-600 dark:text-slate-300 max-w-[90%]"><span class="block text-[10px] font-bold text-slate-400 uppercase">AI</span>${data.answer}</div>`;
        chatHistory.scrollTop = chatHistory.scrollHeight;
    } catch (error) {
        chatHistory.innerHTML += `<div class="bg-red-50 dark:bg-red-900/30 p-2 rounded-lg rounded-tl-none self-start text-red-600 dark:text-red-300 max-w-[90%]"><span class="block text-[10px] font-bold text-red-400 uppercase">AI</span>Sorry, I couldn't process that. Please try again.</div>`;
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
});

// --- 3. INIT ---
document.addEventListener('DOMContentLoaded', () => {
    try {
        initThreeJS();
        runEntryAnimations();
        fetchHistory();
        
        // Init Tilt
        VanillaTilt.init(document.querySelectorAll("[data-tilt]"), {
            max: 5,
            speed: 400,
            glare: true,
            "max-glare": 0.2,
        });
    } catch (e) {
        console.error("Init error:", e);
    }
});
