// Gestion du drag and drop
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('cvFile');
const fileName = document.getElementById('fileName');
const selectedFile = document.getElementById('selectedFile');
const submitButton = document.getElementById('submitButton');
const uploadForm = document.getElementById('uploadForm');
const processingSection = document.getElementById('processingSection');
const resultSection = document.getElementById('resultSection');
const progressFill = document.getElementById('progressFill');
const processingLogs = document.getElementById('processingLogs');

// Événements drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelection(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
    }
});

function handleFileSelection(file) {
    if (file.type !== 'application/pdf') {
        showError('Veuillez sélectionner un fichier PDF');
        return;
    }
    
    if (file.size > 16 * 1024 * 1024) {
        showError('Le fichier est trop volumineux (max 16MB)');
        return;
    }
    
    fileName.textContent = file.name;
    selectedFile.style.display = 'flex';
    submitButton.disabled = false;
    fileInput.files = new DataTransfer().files; // Reset files
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
}

function removeFile() {
    fileInput.value = '';
    selectedFile.style.display = 'none';
    submitButton.disabled = true;
}

function showError(message) {
    alert(message); // Vous pouvez remplacer par un toast plus joli
}

// Soumission du formulaire
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!fileInput.files.length) {
        showError('Veuillez sélectionner un fichier');
        return;
    }
    
    // Afficher la section de traitement
    processingSection.style.display = 'block';
    uploadForm.style.display = 'none';
    
    startProgressAnimation();

    const formData = new FormData();
    formData.append('cv_file', fileInput.files[0]);
    
    try {
        const response = await fetch('/api/process-cv', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();

        stopProgressAnimation();
        
        if (result.success) {
            displayResults(result);
        } else {
            showError(result.error || 'Une erreur est survenue');
            resetForm();
        }
        
    } catch (error) {
        stopProgressAnimation();
        showError('Erreur de connexion: ' + error.message);
        resetForm();
    }
});

function updateProgress(step) {
    const progress = (step / 5) * 100;
    progressFill.style.width = progress + '%';
    
    // Mettre à jour les étapes
    const steps = document.querySelectorAll('.step');
    steps.forEach((s, index) => {
        if (index < step) {
            s.classList.add('active');
        } else {
            s.classList.remove('active');
        }
    });
}

function addLog(message) {
    const logEntry = document.createElement('div');
    logEntry.textContent = '> ' + message;
    logEntry.style.color = '#a78bfa';
    processingLogs.appendChild(logEntry);
    processingLogs.scrollTop = processingLogs.scrollHeight;
}

function displayResults(data) {
    processingSection.style.display = 'none';
    resultSection.style.display = 'block';
    
    resultSection.innerHTML = `
        <div class="result-header" style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: var(--success-color); font-size: 2rem;">
                <i class="fas fa-check-circle"></i> CV Optimisé avec Succès !
            </h2>
            <p style="color: var(--text-light);">Votre CV a été amélioré pour correspondre parfaitement aux attentes des recruteurs</p>
        </div>
        
        <div class="result-card">
            <h3><i class="fas fa-user"></i> Informations Candidat</h3>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Nom Complet</div>
                    <div class="info-value">${data.candidate_info.nom}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Email</div>
                    <div class="info-value">${data.candidate_info.email}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Téléphone</div>
                    <div class="info-value">${data.candidate_info.telephone}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Poste Actuel</div>
                    <div class="info-value">${data.candidate_info.poste}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Localisation</div>
                    <div class="info-value">${data.candidate_info.localisation}</div>
                </div>
            </div>
        </div>
        
        <div class="result-card">
            <h3><i class="fas fa-briefcase"></i> Offre Recommandée</h3>
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">Titre du Poste</div>
                    <div class="info-value">${data.job_info.titre}</div>
                </div>

                <div class="info-item">
                    <div class="info-label">Lien</div>
                    <div class="info-value">
                        <a href="${data.job_info.lien}" target="_blank" style="color: var(--primary-color);">
                            Voir l'offre <i class="fas fa-external-link-alt"></i>
                        </a>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="result-card">
            <h3><i class="fas fa-chart-bar"></i> Scores de Matching</h3>
            <div class="scores-grid">
                <div class="score-card">
                    <div class="score-value">${data.scores.matching}%</div>
                    <div class="score-label">Score Matching</div>
                </div>
            
            </div>
        </div>
        
        <div class="result-card">
            <h3><i class="fas fa-cogs"></i> Compétences Améliorées</h3>
            <div class="skills-list">
                ${data.competences_ameliorees.map(skill => `
                    <div class="skill-item">
                        <i class="fas fa-check"></i>
                        ${skill}
                    </div>
                `).join('')}
            </div>
        </div>
        
        <div class="result-card">
            <h3><i class="fas fa-history"></i> Expériences Optimisées</h3>
            <div class="experience-list">
                ${data.experiences_ameliorees.map(exp => `
                    <div class="experience-item">
                        <i class="fas fa-bullseye"></i>
                        ${exp}
                    </div>
                `).join('')}
            </div>
        </div>
        
        <div class="download-buttons">
            <a href="#" class="download-btn primary" onclick="downloadFile('${data.fichiers_generes.pdf}')">
                <i class="fas fa-file-pdf"></i>
                Télécharger le PDF
            </a>
            <a href="#" class="download-btn" onclick="downloadFile('${data.fichiers_generes.latex}')">
                <i class="fas fa-code"></i>
                Télécharger LaTeX
            </a>
            <button class="download-btn" onclick="resetForm()">
                <i class="fas fa-redo"></i>
                Nouveau CV
            </button>
        </div>
    `;
}

function downloadFile(filename) {
    if (filename) {
        window.open(`/api/download/${filename}`, '_blank');
    } else {
        showError('Fichier non disponible');
    }
}

function resetForm() {
    resultSection.style.display = 'none';
    processingSection.style.display = 'none';
    uploadForm.style.display = 'block';
    removeFile();
    progressFill.style.width = '0%';
    processingLogs.innerHTML = '';
    
    // Reset steps
    document.querySelectorAll('.step').forEach((step, index) => {
        if (index === 0) {
            step.classList.add('active');
        } else {
            step.classList.remove('active');
        }
    });
}

// Simulation de progression (à adapter avec WebSockets ou polling)
function simulateProgress() {
    let step = 1;
    const interval = setInterval(() => {
        updateProgress(step);
        step++;
        if (step > 5) {
            clearInterval(interval);
        }
    }, 4000);

}

// Variables pour gérer la progression
let currentStep = 1;
const totalSteps = 5;
let progressInterval;

// Fonction pour mettre à jour la progression
function updateProgress(step) {
    currentStep = step;
    
    // Mettre à jour les bulles
    const steps = document.querySelectorAll('.step');
    steps.forEach((s, index) => {
        if (index + 1 <= step) {
            s.classList.add('active');
        } else {
            s.classList.remove('active');
        }
    });
    
    // Mettre à jour la barre de progression
    const progressPercentage = (step / totalSteps) * 100;
    document.getElementById('progressFill').style.width = progressPercentage + '%';
    
    // Ajouter des logs selon l'étape
    addStepLog(step);
}

// Fonction pour ajouter des logs selon l'étape
function addStepLog(step) {
    const logs = document.getElementById('processingLogs');
    const logEntry = document.createElement('div');
    
    const stepMessages = {
        1: "📄 Extraction du texte du PDF en cours...",
        2: "🔍 Recherche d'offres d'emploi pertinentes...", 
        3: "🎯 Analyse de matching avec l'IA Groq...",
        4: "✨ Enhancement contextuel du CV...",
        5: "📝 Génération du CV amélioré..."
    };
    
    logEntry.textContent = '> ' + stepMessages[step];
    logEntry.style.color = '#a78bfa';
    logs.appendChild(logEntry);
    logs.scrollTop = logs.scrollHeight;
}

// Fonction pour démarrer la progression automatique
function startProgressAnimation() {
    currentStep = 1;
    updateProgress(1);
    
    // Simulation de progression (à adapter avec les vraies étapes)
    progressInterval = setInterval(() => {
        if (currentStep < totalSteps) {
            updateProgress(currentStep + 1);
        } else {
            clearInterval(progressInterval);
        }
    }, 3000); // Change d'étape toutes les 3 secondes
}

// Fonction pour arrêter la progression
function stopProgressAnimation() {
    if (progressInterval) {
        clearInterval(progressInterval);
    }
}

// Fonction pour synchroniser avec les vraies étapes du backend
function updateRealProgress(step, message) {
    updateProgress(step);
    const logs = document.getElementById('processingLogs');
    const logEntry = document.createElement('div');
    logEntry.textContent = '> ' + message;
    logEntry.style.color = '#10b981'; // Vert pour les vraies étapes
    logs.appendChild(logEntry);
    logs.scrollTop = logs.scrollHeight;
}