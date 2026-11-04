/* ==========================================================================
   VIREO CLINICAL — script.js
   Full-stack Frontend Client for Multimodal Skin Cancer Doctor Portal.
   Integrates with FastAPI backend for 3 AI models, SQLite persistence, and PDF reports.
   ========================================================================== */

const STORAGE = {
  SESSION: 'vireo_session',
  REMEMBER: 'vireo_remember_email',
  LIVE_RESULT: 'vireo_live_result',
  LIVE_IMAGE: 'vireo_live_image',
  LIVE_PATIENT: 'vireo_live_patient',
  PREDICT_PREFILL: 'vireo_predict_prefill',
};

function dataURLtoFile(dataurl, filename) {
  try {
    if (!dataurl || typeof dataurl !== 'string') return null;
    const arr = dataurl.split(',');
    if (arr.length < 2) return null;
    const mimeMatch = arr[0].match(/:(.*?);/);
    const mime = mimeMatch ? mimeMatch[1] : 'image/jpeg';
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename || 'retained_lesion.jpg', { type: mime });
  } catch (e) {
    return null;
  }
}

const API_BASE = (window.SKIN_API_BASE || 'http://127.0.0.1:8000').replace(/\/+$/, '');

/* ---------------------------------------------------------------------- */
/* Auth & Session Helpers                                                 */
/* ---------------------------------------------------------------------- */
function getSession() {
  try {
    const s = sessionStorage.getItem(STORAGE.SESSION) || localStorage.getItem(STORAGE.SESSION);
    return s ? JSON.parse(s) : null;
  } catch (e) {
    return null;
  }
}

function setSession(doctor, remember) {
  const payload = JSON.stringify(doctor);
  if (remember) {
    localStorage.setItem(STORAGE.SESSION, payload);
  } else {
    sessionStorage.setItem(STORAGE.SESSION, payload);
  }
}

function clearSession() {
  sessionStorage.removeItem(STORAGE.SESSION);
  localStorage.removeItem(STORAGE.SESSION);
}

function requireAuth() {
  if (!getSession()) {
    window.location.href = 'index.html';
  }
}

function logout() {
  clearSession();
  window.location.href = 'index.html';
}

/* ---------------------------------------------------------------------- */
/* UI Helpers & Toast Notification                                        */
/* ---------------------------------------------------------------------- */
function toast(message, type = 'info') {
  let el = document.querySelector('.toast');
  if (!el) {
    el = document.createElement('div');
    el.className = 'toast';
    document.body.appendChild(el);
  }
  el.className = 'toast ' + type;
  el.textContent = message;
  requestAnimationFrame(() => el.classList.add('show'));
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('show'), 3500);
}

function pct(n) {
  if (typeof n !== 'number' || isNaN(n)) return '0.0%';
  return (n * 100).toFixed(1) + '%';
}

function riskLabel(risk) {
  const map = { low: 'Low Risk', moderate: 'Moderate Risk', high: 'High Risk' };
  return map[String(risk).toLowerCase()] || String(risk);
}

function setFieldError(fieldEl, message) {
  if (!fieldEl) return;
  fieldEl.classList.add('has-error');
  const err = fieldEl.querySelector('.error span') || fieldEl.querySelector('.error');
  if (err) err.textContent = message;
}

function clearFieldError(fieldEl) {
  if (!fieldEl) return;
  fieldEl.classList.remove('has-error');
}

function fieldOf(input) {
  return input ? input.closest('.field') : null;
}

function isValidEmail(v) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(v || '').trim());
}

function initPasswordToggles() {
  document.querySelectorAll('.toggle-pw').forEach((btn) => {
    btn.addEventListener('click', () => {
      const input = btn.parentElement.querySelector('input');
      if (!input) return;
      const isText = input.type === 'text';
      input.type = isText ? 'password' : 'text';
      btn.innerHTML = isText ? eyeIcon() : eyeOffIcon();
    });
  });
}

function eyeIcon() {
  return '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg>';
}

function eyeOffIcon() {
  return '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.94 17.94A10.94 10.94 0 0112 20c-7 0-11-7-11-7a21.6 21.6 0 015.06-6.06M9.9 4.24A10.6 10.6 0 0112 4c7 0 11 7 11 7a21.6 21.6 0 01-2.61 3.68"/><path d="M14.12 14.12a3 3 0 11-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>';
}

/* ---------------------------------------------------------------------- */
/* App Shell & Navigation Initialization                                  */
/* ---------------------------------------------------------------------- */
function initAppShell() {
  const session = getSession();
  if (session) {
    document.querySelectorAll('[data-doctor-name]').forEach((el) => (el.textContent = session.fullName));
    document.querySelectorAll('[data-doctor-role]').forEach((el) => (el.textContent = session.specialization));
    document.querySelectorAll('[data-doctor-initials]').forEach((el) => {
      const parts = session.fullName.replace('Dr.', '').trim().split(' ').filter(Boolean);
      const initials = parts.map((s) => s[0]).join('').slice(0, 2).toUpperCase() || 'DR';
      el.textContent = initials;
    });
    document.querySelectorAll('[data-doctor-email]').forEach((el) => (el.textContent = session.email));
    document.querySelectorAll('[data-doctor-phone]').forEach((el) => (el.textContent = session.phone));
    document.querySelectorAll('[data-doctor-reg]').forEach((el) => (el.textContent = session.regNumber));
    document.querySelectorAll('[data-doctor-spec]').forEach((el) => (el.textContent = session.specialization));
    document.querySelectorAll('[data-doctor-hospital]').forEach((el) => (el.textContent = session.hospital));
  }

  const dateEl = document.querySelector('[data-current-date]');
  if (dateEl) {
    dateEl.textContent = new Date().toLocaleDateString('en-US', {
      weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
    });
  }

  document.querySelectorAll('[data-logout]').forEach((el) => el.addEventListener('click', logout));

  const menuToggle = document.querySelector('.menu-toggle');
  const sidebar = document.querySelector('.sidebar');
  if (menuToggle && sidebar) {
    menuToggle.addEventListener('click', () => sidebar.classList.toggle('open'));
  }
}

/* ---------------------------------------------------------------------- */
/* LOGIN PAGE                                                             */
/* ---------------------------------------------------------------------- */
function initLoginPage() {
  const form = document.getElementById('loginForm');
  if (!form) return;

  const remembered = localStorage.getItem(STORAGE.REMEMBER);
  if (remembered) {
    const em = document.getElementById('loginEmail');
    const rm = document.getElementById('rememberMe');
    if (em) em.value = remembered;
    if (rm) rm.checked = true;
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const emailInput = document.getElementById('loginEmail');
    const passInput = document.getElementById('loginPassword');

    clearFieldError(fieldOf(emailInput));
    clearFieldError(fieldOf(passInput));

    const email = emailInput.value.trim();
    const pass = passInput.value;

    let valid = true;
    if (!email) {
      setFieldError(fieldOf(emailInput), 'Email is required.');
      valid = false;
    } else if (!isValidEmail(email)) {
      setFieldError(fieldOf(emailInput), 'Enter a valid email address.');
      valid = false;
    }
    if (!pass) {
      setFieldError(fieldOf(passInput), 'Password is required.');
      valid = false;
    }
    if (!valid) return;

    try {
      const resp = await fetch(API_BASE + '/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password: pass }),
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        setFieldError(fieldOf(passInput), errData.detail || 'Incorrect email or password.');
        return;
      }

      const res = await resp.json();
      const remember = document.getElementById('rememberMe')?.checked;
      if (remember) {
        localStorage.setItem(STORAGE.REMEMBER, email);
      } else {
        localStorage.removeItem(STORAGE.REMEMBER);
      }

      setSession(res.doctor, remember);
      toast('Welcome back, ' + res.doctor.fullName.split(' ')[0] + '.', 'success');
      setTimeout(() => (window.location.href = 'dashboard.html'), 400);
    } catch (err) {
      // Local fallback for offline demonstration if server is starting
      if (email.toLowerCase() === 'meera.kulkarni@vireoclinic.in' && pass === 'Demo@1234') {
        const doc = {
          fullName: 'Dr. Meera Kulkarni',
          regNumber: 'MH-DERM-88214',
          email: 'meera.kulkarni@vireoclinic.in',
          phone: '+91 98220 11223',
          specialization: 'Dermatology & Oncology',
          hospital: 'Sunrise Multispecialty Hospital, Aurangabad',
        };
        setSession(doc, false);
        toast('Welcome back (Local Session).', 'success');
        setTimeout(() => (window.location.href = 'dashboard.html'), 400);
      } else {
        setFieldError(fieldOf(passInput), 'Unable to connect to authentication service.');
      }
    }
  });
}

/* ---------------------------------------------------------------------- */
/* SIGNUP PAGE                                                            */
/* ---------------------------------------------------------------------- */
function initSignupPage() {
  const form = document.getElementById('signupForm');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fields = {
      fullName: document.getElementById('suFullName'),
      regNumber: document.getElementById('suRegNumber'),
      email: document.getElementById('suEmail'),
      phone: document.getElementById('suPhone'),
      specialization: document.getElementById('suSpecialization'),
      hospital: document.getElementById('suHospital'),
      password: document.getElementById('suPassword'),
      confirm: document.getElementById('suConfirm'),
    };
    const terms = document.getElementById('suTerms');

    Object.values(fields).forEach((f) => clearFieldError(fieldOf(f)));
    clearFieldError(fieldOf(terms));

    let valid = true;
    if (!fields.fullName.value.trim()) { setFieldError(fieldOf(fields.fullName), 'Full name is required.'); valid = false; }
    if (!fields.regNumber.value.trim()) { setFieldError(fieldOf(fields.regNumber), 'Medical registration number is required.'); valid = false; }
    if (!fields.email.value.trim()) {
      setFieldError(fieldOf(fields.email), 'Email is required.'); valid = false;
    } else if (!isValidEmail(fields.email.value)) {
      setFieldError(fieldOf(fields.email), 'Enter a valid email address.'); valid = false;
    }
    if (!fields.phone.value.trim()) { setFieldError(fieldOf(fields.phone), 'Phone number is required.'); valid = false; }
    if (!fields.specialization.value.trim()) { setFieldError(fieldOf(fields.specialization), 'Specialization is required.'); valid = false; }
    if (!fields.hospital.value.trim()) { setFieldError(fieldOf(fields.hospital), 'Hospital name is required.'); valid = false; }
    if (fields.password.value.length < 6) { setFieldError(fieldOf(fields.password), 'Password must be at least 6 characters.'); valid = false; }
    if (fields.confirm.value !== fields.password.value) { setFieldError(fieldOf(fields.confirm), 'Passwords do not match.'); valid = false; }
    if (!terms.checked) { setFieldError(fieldOf(terms), 'You must accept the terms.'); valid = false; }

    if (!valid) return;

    try {
      const resp = await fetch(API_BASE + '/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          fullName: fields.fullName.value.trim(),
          regNumber: fields.regNumber.value.trim(),
          email: fields.email.value.trim(),
          phone: fields.phone.value.trim(),
          specialization: fields.specialization.value.trim(),
          hospital: fields.hospital.value.trim(),
          password: fields.password.value,
        }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        setFieldError(fieldOf(fields.email), err.detail || 'Could not register doctor account.');
        return;
      }

      toast('Doctor account created. Please log in.', 'success');
      setTimeout(() => (window.location.href = 'index.html'), 800);
    } catch (err) {
      toast('Network error during registration.', 'error');
    }
  });
}

/* ---------------------------------------------------------------------- */
/* DASHBOARD PAGE                                                         */
/* ---------------------------------------------------------------------- */
async function initDashboardPage() {
  const statTotal = document.getElementById('statTotalPatients');
  if (!statTotal) return;

  try {
    const statsResp = await fetch(API_BASE + '/stats');
    if (statsResp.ok) {
      const s = await statsResp.json();
      document.getElementById('statTotalPatients').textContent = s.total_patients;
      document.getElementById('statTodayPatients').textContent = s.today_patients;
      document.getElementById('statTotalPredictions').textContent = s.total_predictions;
      document.getElementById('statHighRisk').textContent = s.high_risk_cases;
    }
  } catch (e) {
    console.warn('Could not fetch stats from API:', e);
  }

  const tbody = document.getElementById('recentPatientsBody');
  if (tbody) {
    try {
      const resp = await fetch(API_BASE + '/patients');
      if (resp.ok) {
        const patients = await resp.json();
        const recent = patients.slice(0, 5);
        if (recent.length === 0) {
          tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:30px; color:var(--muted);">No patients recorded yet.</td></tr>';
        } else {
          tbody.innerHTML = recent.map((p) => {
            const risk = p.latest_risk || 'low';
            const pred = p.latest_prediction || 'Screening pending';
            const dt = (p.latest_report_date || p.created_at || '').slice(0, 10);
            return `<tr>
              <td class="pid">${p.patient_id}</td>
              <td class="name">${p.name}</td>
              <td>${p.age}</td>
              <td class="mono">${dt}</td>
              <td><span class="badge ${risk}">${riskLabel(risk)}</span></td>
              <td>${pred}</td>
              <td><a href="patients.html?view=${encodeURIComponent(p.patient_id)}" class="btn btn-ghost btn-sm">Patient History →</a></td>
            </tr>`;
          }).join('');
        }
      }
    } catch (e) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:30px; color:var(--muted);">Backend offline. Start the FastAPI service.</td></tr>';
    }
  }
}

/* ---------------------------------------------------------------------- */
/* PATIENTS PAGE                                                          */
/* ---------------------------------------------------------------------- */
async function initPatientsPage() {
  const tbody = document.getElementById('patientsBody');
  if (!tbody) return;

  const searchInput = document.getElementById('patientSearch');
  const riskFilter = document.getElementById('riskFilter');
  const dateFilter = document.getElementById('dateFilter');

  async function loadPatients() {
    try {
      const params = new URLSearchParams();
      if (searchInput && searchInput.value.trim()) params.set('search', searchInput.value.trim());
      if (riskFilter && riskFilter.value !== 'all') params.set('risk', riskFilter.value);
      if (dateFilter && dateFilter.value) params.set('date', dateFilter.value);

      const resp = await fetch(API_BASE + '/patients?' + params.toString());
      if (!resp.ok) throw new Error('API error');
      const patients = await resp.json();

      if (!patients.length) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:30px; color:var(--muted);">No patients matching your search criteria.</td></tr>';
        return;
      }

      tbody.innerHTML = patients.map((p) => {
        const risk = p.latest_risk || 'low';
        const pred = p.latest_prediction || 'Assessment pending';
        const dt = (p.latest_report_date || p.created_at || '').slice(0, 10);
        return `<tr>
          <td class="pid">${p.patient_id}</td>
          <td class="name"><strong>${p.name}</strong></td>
          <td>${p.age} yrs (${p.gender})</td>
          <td class="mono">${dt}</td>
          <td><span class="badge ${risk}">${riskLabel(risk)}</span></td>
          <td>${pred}</td>
          <td style="white-space:nowrap;">
            <a href="predict.html?pid=${encodeURIComponent(p.patient_id)}&name=${encodeURIComponent(p.name)}&age=${p.age}&gender=${encodeURIComponent(p.gender)}" class="btn btn-primary btn-sm" title="Run multimodal prediction">+ Predict</a>
            <button class="btn btn-outline btn-sm" onclick="openPatientHistoryModal('${p.patient_id}')">Reports (${p.patient_id})</button>
          </td>
        </tr>`;
      }).join('');
    } catch (err) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:30px; color:var(--muted);">Failed to load patients from database.</td></tr>';
    }
  }

  if (searchInput) searchInput.addEventListener('input', debounce(loadPatients, 250));
  if (riskFilter) riskFilter.addEventListener('change', loadPatients);
  if (dateFilter) dateFilter.addEventListener('change', loadPatients);

  loadPatients();

  // If URL has ?view=PT-XXXX, automatically open that patient's history modal
  const urlParams = new URLSearchParams(window.location.search);
  const viewPid = urlParams.get('view');
  if (viewPid) {
    setTimeout(() => openPatientHistoryModal(viewPid), 400);
  }
}

function debounce(fn, ms) {
  let timer;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}

/* Patient History Modal */
window.openPatientHistoryModal = async function (patientId) {
  let modal = document.getElementById('patientHistoryModal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'patientHistoryModal';
    modal.className = 'modal-backdrop';
    modal.innerHTML = `
      <div class="modal-card">
        <div class="modal-header">
          <h3 id="modalPatientTitle">Patient Report History</h3>
          <button type="button" class="close-btn" onclick="document.getElementById('patientHistoryModal').classList.remove('open')">&times;</button>
        </div>
        <div id="modalPatientContent" style="padding:20px; max-height:420px; overflow-y:auto;">Loading history…</div>
        <div class="modal-footer" style="padding:14px 20px; border-top:1px solid var(--border); text-align:right;">
          <button type="button" class="btn btn-outline btn-sm" onclick="document.getElementById('patientHistoryModal').classList.remove('open')">Close</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);
  }

  const titleEl = document.getElementById('modalPatientTitle');
  const contentEl = document.getElementById('modalPatientContent');
  titleEl.textContent = `Patient History · ${patientId}`;
  contentEl.innerHTML = '<p class="text-muted">Fetching historical reports from backend…</p>';
  modal.classList.add('open');

  try {
    const resp = await fetch(API_BASE + `/patients/${encodeURIComponent(patientId)}`);
    if (!resp.ok) throw new Error('Patient not found');
    const data = await resp.json();
    const p = data.patient;
    const reports = data.reports || [];

    let html = `<div style="margin-bottom:16px; padding-bottom:12px; border-bottom:1px solid var(--border);">
      <div><strong>${p.name}</strong> (${p.age} years, ${p.gender})</div>
      <div class="text-muted" style="font-size:12px;">Phone: ${p.phone || 'N/A'} · Email: ${p.email || 'N/A'}</div>
    </div>`;

    if (reports.length === 0) {
      html += `<div style="text-align:center; padding:30px; color:var(--muted);">
        <p>No historical reports recorded for this patient yet.</p>
        <a href="predict.html?pid=${encodeURIComponent(p.patient_id)}&name=${encodeURIComponent(p.name)}&age=${p.age}&gender=${encodeURIComponent(p.gender)}" class="btn btn-primary btn-sm">Run First AI Assessment</a>
      </div>`;
    } else {
      html += '<table style="width:100%; font-size:13px;"><thead><tr><th>Report ID</th><th>Date</th><th>Score</th><th>Result</th><th>Actions</th></tr></thead><tbody>';
      reports.forEach((r) => {
        const dt = (r.created_at || '').slice(0, 19).replace('T', ' ');
        const scorePct = pct(r.ensemble_score);
        html += `<tr>
          <td class="mono"><strong>${r.report_id}</strong></td>
          <td class="mono" style="font-size:11px;">${dt}</td>
          <td><b>${scorePct}</b></td>
          <td><span class="badge ${r.risk_level}">${r.prediction_label}</span></td>
          <td style="white-space:nowrap;">
            <a href="results.html?report_id=${encodeURIComponent(r.report_id)}" class="btn btn-ghost btn-sm">View</a>
            <a href="${API_BASE}/reports/${encodeURIComponent(r.report_id)}/pdf" class="btn btn-accent btn-sm" download>Download PDF</a>
          </td>
        </tr>`;
      });
      html += '</tbody></table>';
    }
    contentEl.innerHTML = html;
  } catch (err) {
    contentEl.innerHTML = '<p style="color:var(--danger);">Could not load patient reports.</p>';
  }
};

/* ---------------------------------------------------------------------- */
/* REPORTS ARCHIVE PAGE (reports.html)                                    */
/* ---------------------------------------------------------------------- */
async function initReportsPage() {
  const tbody = document.getElementById('reportsTableBody');
  if (!tbody) return;

  const searchInput = document.getElementById('reportSearch');
  const riskFilter = document.getElementById('reportRiskFilter');

  async function loadReports() {
    try {
      const resp = await fetch(API_BASE + '/reports');
      if (!resp.ok) throw new Error('API error');
      let reports = await resp.json();

      const q = (searchInput?.value || '').toLowerCase().trim();
      const risk = riskFilter?.value || 'all';

      if (q) {
        reports = reports.filter((r) =>
          (r.report_id && r.report_id.toLowerCase().includes(q)) ||
          (r.patient_name && r.patient_name.toLowerCase().includes(q)) ||
          (r.patient_id && r.patient_id.toLowerCase().includes(q))
        );
      }
      if (risk !== 'all') {
        reports = reports.filter((r) => (r.risk_level || '').toLowerCase() === risk.toLowerCase());
      }

      if (!reports.length) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:30px; color:var(--muted);">No reports found matching your filter.</td></tr>';
        return;
      }

      tbody.innerHTML = reports.map((r) => {
        const dt = (r.created_at || '').slice(0, 19).replace('T', ' ');
        const effPct = pct(r.efficientnet_score);
        const cliPct = pct(r.clinical_score);
        const isLegacy = (r.resnet50_score != null && r.ensemble_score != null);
        const scoreDisplay = isLegacy
          ? `Eff: ${effPct} · XGB: ${cliPct} <div class="text-muted" style="font-size:10px;">Archived Ens: ${pct(r.ensemble_score)}</div>`
          : `Eff: ${effPct} · XGB: ${cliPct}`;
        const effHigh = (r.efficientnet_score || 0) >= 0.5;
        const cliHigh = (r.clinical_score || 0) >= 0.5;
        const assessmentDisplay = r.prediction_label || (effHigh || cliHigh ? 'Elevated Risk' : 'Low-Risk');
        const badgeClass = (effHigh || cliHigh) ? 'high' : 'low';
        const badgeText = isLegacy ? 'Archived (3-Model)' : 'Dual-Model';

        return `<tr>
          <td class="mono"><strong>${r.report_id}</strong></td>
          <td>
            <strong>${r.patient_name || 'Patient'}</strong>
            <div class="text-muted mono" style="font-size:11px;">${r.patient_id}</div>
          </td>
          <td class="mono">${dt}</td>
          <td>${scoreDisplay}</td>
          <td>${assessmentDisplay}</td>
          <td><span class="badge ${badgeClass}">${badgeText}</span></td>
          <td style="text-align:right; white-space:nowrap;">
            <a href="results.html?report_id=${encodeURIComponent(r.report_id)}" class="btn btn-ghost btn-sm">View Result</a>
            <a href="${API_BASE}/reports/${encodeURIComponent(r.report_id)}/pdf" class="btn btn-accent btn-sm" download>Download PDF</a>
          </td>
        </tr>`;
      }).join('');
    } catch (err) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding:30px; color:var(--muted);">Failed to load reports archive from backend.</td></tr>';
    }
  }

  if (searchInput) searchInput.addEventListener('input', debounce(loadReports, 200));
  if (riskFilter) riskFilter.addEventListener('change', loadReports);

  loadReports();
}

/* ---------------------------------------------------------------------- */
/* PATIENT ASSESSMENT PAGE (patient.html)                                  */
/* ---------------------------------------------------------------------- */
function initPatientFormPage() {
  const form = document.getElementById('patientForm');
  if (!form) return;

  // Toggle pill groups (Yes/No)
  document.querySelectorAll('.pill-toggle').forEach((group) => {
    group.querySelectorAll('input[type="radio"]').forEach((input) => {
      input.addEventListener('change', () => {
        group.querySelectorAll('label').forEach((l) => l.classList.remove('active'));
        if (input.nextElementSibling) input.nextElementSibling.classList.add('active');
      });
    });
  });

  // Lesion Location Chips
  document.querySelectorAll('.chip-grid .chip input').forEach((input) => {
    input.addEventListener('change', () => {
      document.querySelectorAll('.chip-grid .chip').forEach((c) => c.classList.remove('active'));
      input.closest('.chip')?.classList.add('active');
    });
  });

  // UV Exposure Slider
  const uvSlider = document.getElementById('uvExposure');
  const uvVal = document.getElementById('uvExposureVal');
  if (uvSlider && uvVal) {
    const labels = ['Minimal', 'Low', 'Moderate', 'High', 'Very High'];
    const update = () => (uvVal.textContent = labels[uvSlider.value] || 'Moderate');
    uvSlider.addEventListener('input', update);
    update();
  }

  // Image upload & dropzone
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('lesionImageInput');
  const browseBtn = document.getElementById('patientBrowseBtn');
  const previewWrap = document.getElementById('imagePreviewWrap');
  const previewImg = document.getElementById('imagePreviewImg');
  const previewActions = document.getElementById('patientPreviewActions');
  const replaceBtn = document.getElementById('patientReplaceBtn');
  const removeBtn = document.getElementById('removeImageBtn');
  const fileNameEl = document.getElementById('imgFileName');
  let selectedImageFile = null;

  function handleFile(file) {
    if (!file) return;
    const name = (file.name || '').toLowerCase();
    const isAllowedExt = name.endsWith('.jpg') || name.endsWith('.jpeg') || name.endsWith('.png');
    const mime = (file.type || '').toLowerCase();
    const isAllowedMime = !mime || ['image/jpeg', 'image/jpg', 'image/png', 'image/pjpeg', 'image/x-png'].includes(mime);
    if (!isAllowedExt && !isAllowedMime) {
      toast('Unsupported format. Please upload JPG or PNG.', 'error');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      toast('Image exceeds the 10MB limit.', 'error');
      return;
    }
    selectedImageFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      fileNameEl.textContent = file.name;
      dropzone.style.display = 'none';
      previewWrap.style.display = 'block';
      if (previewActions) previewActions.style.display = 'flex';
    };
    reader.readAsDataURL(file);
  }

  if (browseBtn) {
    browseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }

  if (dropzone && fileInput) {
    dropzone.addEventListener('click', (e) => {
      if (e.target === fileInput || e.target === browseBtn) return;
      fileInput.click();
    });
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
      if (e.dataTransfer && e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.value = '';
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length) handleFile(fileInput.files[0]);
    });
    if (previewImg) {
      previewImg.addEventListener('click', () => fileInput.click());
    }
    if (replaceBtn) {
      replaceBtn.addEventListener('click', () => fileInput.click());
    }
    if (removeBtn) {
      removeBtn.addEventListener('click', () => {
        selectedImageFile = null;
        fileInput.value = '';
        previewWrap.style.display = 'none';
        if (previewActions) previewActions.style.display = 'none';
        dropzone.style.display = 'block';
      });
    }
  }

  // Submit Handler -> Real 3-Model AI Prediction & SQLite persistence
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    let valid = true;

    // Validate required fields
    form.querySelectorAll('[data-required="true"]').forEach((input) => {
      const field = fieldOf(input);
      clearFieldError(field);
      if (!input.value || !input.value.trim()) {
        setFieldError(field, 'This field is required.');
        valid = false;
      }
    });

    const emailInput = document.getElementById('patEmail');
    if (emailInput && emailInput.value.trim() && !isValidEmail(emailInput.value)) {
      setFieldError(fieldOf(emailInput), 'Enter a valid email address.');
      valid = false;
    }

    if (!selectedImageFile) {
      toast('Please upload a skin lesion image before analysis.', 'error');
      valid = false;
    }

    if (!valid) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      return;
    }

    const submitBtn = form.querySelector('button[type="submit"]');
    const origText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Analyzing Lesion (3 AI Models)…';

    try {
      // 1. Map form inputs to the verified 21 clinical fields in feature_schema.json
      const locationRadio = form.querySelector('input[name="lesionLocation"]:checked');
      const regionVal = (locationRadio ? locationRadio.value : 'FACE').toUpperCase();
      // Map region to exact vocabulary in feature_schema.json
      const validRegions = ["ABDOMEN", "ARM", "BACK", "CHEST", "EAR", "FACE", "FOOT", "FOREARM", "HAND", "LIP", "NECK", "NOSE", "SCALP", "THIGH"];
      const matchedRegion = validRegions.includes(regionVal) ? regionVal : (regionVal === 'LEG' ? 'THIGH' : 'FACE');

      const fd = new FormData();
      fd.append('image', selectedImageFile, selectedImageFile.name);
      fd.append('age', document.getElementById('patAge').value.trim());
      fd.append('region', matchedRegion);

      // Gender
      const genderVal = (document.getElementById('patGender').value || '').toUpperCase();
      if (['MALE', 'FEMALE'].includes(genderVal)) fd.append('gender', genderVal);

      // Symptoms
      const getYesNo = (name) => {
        const rad = form.querySelector(`input[name="${name}"]:checked`);
        return rad && rad.value === 'Yes' ? 'TRUE' : 'FALSE';
      };
      fd.append('changed', getYesNo('changed'));
      fd.append('itch', getYesNo('itching'));
      fd.append('bleed', getYesNo('bleeding'));
      fd.append('hurt', getYesNo('pain'));
      fd.append('skin_cancer_history', getYesNo('previousHistory'));
      fd.append('cancer_history', getYesNo('familyHistory'));

      // 2. Run real prediction on backend
      const predResp = await fetch(API_BASE + '/predict', {
        method: 'POST',
        body: fd,
      });

      if (!predResp.ok) {
        const err = await predResp.json().catch(() => ({}));
        throw new Error(err.detail || 'Prediction failed on backend.');
      }

      const predData = await predResp.json();

      // 3. Persist patient to database
      const patIdVal = document.getElementById('patId').value.trim() ||
        ('PT-' + Math.floor(10000 + Math.random() * 89999));
      const patientPayload = {
        patient_id: patIdVal,
        name: document.getElementById('patName').value.trim(),
        age: parseInt(document.getElementById('patAge').value.trim(), 10),
        gender: document.getElementById('patGender').value,
        dob: document.getElementById('patDob').value,
        phone: document.getElementById('patPhone').value.trim(),
        email: emailInput.value.trim(),
        address: document.getElementById('patAddress').value.trim(),
      };

      const patSaveResp = await fetch(API_BASE + '/patients', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patientPayload),
      });
      const savedPatient = patSaveResp.ok ? await patSaveResp.json() : patientPayload;

      // 4. Persist report to database
      const reportPayload = {
        patient_id: savedPatient.patient_id,
        patient_name: savedPatient.name,
        patient_age: savedPatient.age,
        patient_gender: savedPatient.gender,
        patient_phone: savedPatient.phone,
        patient_email: savedPatient.email,
        efficientnet_score: predData.models.efficientnet_b0.cancer_score,
        clinical_score: predData.models.clinical_xgboost.cancer_score,
        clinical_inputs: {
          region: matchedRegion,
          changed: getYesNo('changed'),
          itch: getYesNo('itching'),
          bleed: getYesNo('bleeding'),
          hurt: getYesNo('pain'),
          skin_cancer_history: getYesNo('previousHistory'),
          cancer_history: getYesNo('familyHistory'),
          notes: document.getElementById('clinicalNotes').value.trim(),
        },
        doctor_notes: document.getElementById('clinicalNotes').value.trim(),
      };

      const repSaveResp = await fetch(API_BASE + '/reports', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reportPayload),
      });

      const savedReport = repSaveResp.ok ? await repSaveResp.json() : null;
      const targetReportId = savedReport ? savedReport.report_id : '';

      // Cache image preview for results page display
      sessionStorage.setItem(STORAGE.LIVE_IMAGE, previewImg.src);
      sessionStorage.setItem(STORAGE.LIVE_RESULT, JSON.stringify(predData));
      sessionStorage.setItem(STORAGE.LIVE_PATIENT, JSON.stringify(savedPatient));

      toast('Assessment complete & report saved!', 'success');
      window.location.href = `results.html?report_id=${encodeURIComponent(targetReportId)}`;
    } catch (err) {
      toast(err.message || 'Assessment failed. Check backend status.', 'error');
      submitBtn.disabled = false;
      submitBtn.textContent = origText;
    }
  });
}

/* ---------------------------------------------------------------------- */
/* AI PREDICTION PAGE (predict.html)                                      */
/* ---------------------------------------------------------------------- */
function initPredictPage() {
  const form = document.getElementById('predictForm');
  if (!form) return;

  const dropzone = document.getElementById('predDropzone');
  const fileInput = document.getElementById('predImageInput');
  const browseBtn = document.getElementById('predBrowseBtn');
  const previewWrap = document.getElementById('predPreviewWrap');
  const previewImg = document.getElementById('predPreviewImg');
  const previewActions = document.getElementById('predPreviewActions');
  const replaceBtn = document.getElementById('predReplaceBtn');
  const removeBtn = document.getElementById('predRemoveBtn');
  const fileNameEl = document.getElementById('predFileName');
  const fieldsWrap = document.getElementById('clinicalFields');
  const submitBtn = document.getElementById('predSubmitBtn');
  const statusEl = document.getElementById('predStatus');
  const alertBox = document.getElementById('predAlert');

  let imageFile = null;
  let schema = null;
  let isNewImageSelected = false;

  const urlParams = new URLSearchParams(window.location.search);
  const mode = urlParams.get('mode');
  const reportIdParam = urlParams.get('report_id');
  let prefill = null;
  try {
    prefill = JSON.parse(sessionStorage.getItem(STORAGE.PREDICT_PREFILL) || 'null');
  } catch (e) {}

  // If user navigated directly without mode/report, clear any leftover temporary state
  if (!mode && !reportIdParam) {
    sessionStorage.removeItem(STORAGE.PREDICT_PREFILL);
    prefill = null;
  }

  const infoBanner = document.getElementById('predInfoBanner');
  const infoText = document.getElementById('predInfoText');
  const clearPrefillBtn = document.getElementById('clearPrefillBtn');

  if (clearPrefillBtn) {
    clearPrefillBtn.addEventListener('click', () => {
      sessionStorage.removeItem(STORAGE.PREDICT_PREFILL);
      form.reset();
      imageFile = null;
      isNewImageSelected = false;
      if (fileInput) fileInput.value = '';
      if (previewWrap) previewWrap.style.display = 'none';
      if (previewActions) previewActions.style.display = 'none';
      if (dropzone) dropzone.style.display = 'block';
      if (infoBanner) infoBanner.style.display = 'none';
      toast('Prefill cleared. Form reset.', 'info');
    });
  }

  function showAlert(msg) {
    const t = document.getElementById('predAlertText');
    if (t) t.textContent = msg;
    if (alertBox) alertBox.style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
  function clearAlert() { if (alertBox) alertBox.style.display = 'none'; }
  function setStatus(msg) { if (statusEl) statusEl.textContent = msg || ''; }

  function handleFile(file) {
    if (!file) return;
    const name = (file.name || '').toLowerCase();
    const isAllowedExt = name.endsWith('.jpg') || name.endsWith('.jpeg') || name.endsWith('.png');
    const mime = (file.type || '').toLowerCase();
    const isAllowedMime = !mime || ['image/jpeg', 'image/jpg', 'image/png', 'image/pjpeg', 'image/x-png'].includes(mime);
    if (!isAllowedExt && !isAllowedMime) {
      showAlert('Unsupported image format. Please upload a genuine JPG, JPEG, or PNG image.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showAlert('Image exceeds the 10MB size limit.');
      return;
    }
    clearAlert();
    imageFile = file;
    isNewImageSelected = true;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      fileNameEl.textContent = file.name;
      dropzone.style.display = 'none';
      previewWrap.style.display = 'block';
      if (previewActions) previewActions.style.display = 'flex';
    };
    reader.readAsDataURL(file);

    if (infoBanner && infoText) {
      infoText.textContent = `New lesion image selected: "${file.name}". Clinical inputs preserved. Click "Run AI Assessment" to execute inference.`;
      infoBanner.style.display = 'block';
    }
  }

  // Browse button
  if (browseBtn) {
    browseBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.click();
    });
  }

  // Dropzone click & drag events
  if (dropzone && fileInput) {
    dropzone.addEventListener('click', (e) => {
      if (e.target === fileInput || e.target === browseBtn) return;
      fileInput.click();
    });
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
      if (e.dataTransfer && e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('click', (e) => {
      e.stopPropagation();
      fileInput.value = '';
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files.length) handleFile(fileInput.files[0]);
    });
    if (previewImg) {
      previewImg.addEventListener('click', () => fileInput.click());
    }
    if (replaceBtn) {
      replaceBtn.addEventListener('click', () => fileInput.click());
    }
    if (removeBtn) {
      removeBtn.addEventListener('click', () => {
        imageFile = null;
        isNewImageSelected = false;
        fileInput.value = '';
        previewWrap.style.display = 'none';
        if (previewActions) previewActions.style.display = 'none';
        dropzone.style.display = 'block';
        if (mode === 'change_image' && infoBanner && infoText) {
          infoText.textContent = 'Current image removed. Please upload or select a new skin lesion image.';
        }
      });
    }
  }

  function applyPrefillValue(el, val) {
    if (!el || val === null || val === undefined || val === '') return;
    if (el.tagName === 'SELECT') {
      let strVal = String(val).trim().toUpperCase();
      if (val === true) strVal = 'TRUE';
      if (val === false) strVal = 'FALSE';
      const match = Array.from(el.options).find(
        (o) => o.value.toUpperCase() === strVal || o.value.toLowerCase() === String(val).toLowerCase()
      );
      if (match) {
        el.value = match.value;
      } else {
        el.value = strVal;
      }
    } else {
      el.value = val;
    }
  }

  function renderField(f) {
    const req = f.required ? ' data-required="true"' : '';
    const star = f.required ? ' *' : ' <span class="text-muted">(optional)</span>';

    if (f.options) {
      const opts = f.options.map((o) => `<option value="${o.value}">${o.label}</option>`).join('');
      return `<div class="field">
        <label for="pf_${f.name}">${f.label}${star}</label>
        <select id="pf_${f.name}" data-feature="${f.name}"${req}>
          <option value="">Unknown / not provided</option>
          ${opts}
        </select>
        <div class="error">⚠ <span></span></div>
      </div>`;
    }

    const step = f.type === 'integer' ? '1' : '0.1';
    return `<div class="field">
      <label for="pf_${f.name}">${f.label}${star}</label>
      <input type="number" id="pf_${f.name}" data-feature="${f.name}" min="${f.min}" max="${f.max}" step="${step}" placeholder="e.g. ${f.min}"${req}>
      <div class="error">⚠ <span></span></div>
    </div>`;
  }

  function renderClinicalFields(s, fetchedReport) {
    if (!fieldsWrap) return;
    let html = '<div class="form-section-sub" style="margin:14px 0 8px; font-weight:600;">Measurements & Core Demographics</div>';
    html += '<div class="form-row-3">' + s.numeric_features.map(renderField).join('') + '</div>';

    html += '<div class="form-section-sub" style="margin:20px 0 8px; font-weight:600;">Clinical Symptoms & History</div>';
    const bools = s.boolean_features.map(renderField);
    for (let i = 0; i < bools.length; i += 3) {
      html += '<div class="form-row-3">' + bools.slice(i, i + 3).join('') + '</div>';
    }

    html += '<div class="form-section-sub" style="margin:20px 0 8px; font-weight:600;">Lesion Region & Background</div>';
    const cats = s.categorical_features.map(renderField);
    for (let i = 0; i < cats.length; i += 2) {
      html += '<div class="form-row-split">' + cats.slice(i, i + 2).join('') + '</div>';
    }

    fieldsWrap.innerHTML = html;

    // Prefill from URL query params (pid, name, age, gender) if present
    if (urlParams.get('pid')) document.getElementById('predId').value = urlParams.get('pid');
    if (urlParams.get('name')) document.getElementById('predName').value = urlParams.get('name');
    if (urlParams.get('age')) {
      const ageInput = document.getElementById('pf_age');
      if (ageInput) ageInput.value = urlParams.get('age');
    }
    if (urlParams.get('gender')) {
      const genInput = document.getElementById('pf_gender');
      if (genInput) applyPrefillValue(genInput, urlParams.get('gender'));
    }

    // Determine active clinical inputs from fetchedReport or prefill session
    let activeInputs = null;
    if (fetchedReport) {
      if (fetchedReport.clinical_inputs && typeof fetchedReport.clinical_inputs === 'object') {
        activeInputs = fetchedReport.clinical_inputs;
      } else if (fetchedReport.clinical_inputs_json) {
        try { activeInputs = JSON.parse(fetchedReport.clinical_inputs_json); } catch (e) {}
      }
    }
    if (!activeInputs && prefill && prefill.clinical_inputs) {
      activeInputs = prefill.clinical_inputs;
    }

    if (activeInputs) {
      for (const [key, val] of Object.entries(activeInputs)) {
        if (val === null || val === undefined || val === '') continue;
        const el = form.querySelector(`[data-feature="${key}"]`) || document.getElementById(`pf_${key}`);
        if (el) {
          applyPrefillValue(el, val);
        }
      }
    }

    const activeAge = (fetchedReport && fetchedReport.patient_age != null)
      ? fetchedReport.patient_age
      : (prefill && prefill.age != null && prefill.age !== '' ? prefill.age : null);
    if (activeAge != null) {
      const ageEl = document.getElementById('pf_age');
      if (ageEl) ageEl.value = activeAge;
    }

    const activeGender = (fetchedReport && fetchedReport.patient_gender) || (prefill && prefill.gender);
    if (activeGender) {
      const genEl = document.getElementById('pf_gender');
      if (genEl) applyPrefillValue(genEl, activeGender);
    }
  }

  function setupWorkflowMode(fetchedReport) {
    const activePrefill = prefill || {};
    const patientName = activePrefill.name || (fetchedReport && fetchedReport.patient_name) || '';
    const patientId = activePrefill.patient_id || (fetchedReport && fetchedReport.patient_id) || '';

    if (patientName) document.getElementById('predName').value = patientName;
    if (patientId) document.getElementById('predId').value = patientId;

    const sessionImage = activePrefill.image_data_url || sessionStorage.getItem(STORAGE.LIVE_IMAGE);

    if (mode === 'change_image') {
      isNewImageSelected = false;
      imageFile = null;
      if (sessionImage) {
        previewImg.src = sessionImage;
        fileNameEl.textContent = 'Previous lesion image (Select a new image to replace)';
        previewWrap.style.display = 'block';
        if (previewActions) previewActions.style.display = 'flex';
        dropzone.style.display = 'none';
      } else {
        dropzone.style.display = 'block';
        previewWrap.style.display = 'none';
        if (previewActions) previewActions.style.display = 'none';
      }

      if (infoBanner && infoText) {
        const nameDisplay = patientName ? `patient ${patientName} (${patientId})` : `patient ${patientId || 'record'}`;
        infoText.textContent = `Change Image mode for ${nameDisplay}. All previous clinical values are prefilled. Current lesion image is shown below — click "Choose Different Image" or the preview to select a new image.`;
        infoBanner.style.display = 'block';
      }
    } else if (mode === 'predict_again') {
      if (sessionImage) {
        const file = dataURLtoFile(sessionImage, 'retained_lesion.jpg');
        if (file) {
          imageFile = file;
          isNewImageSelected = true;
          previewImg.src = sessionImage;
          fileNameEl.textContent = 'Retained lesion image from previous assessment';
          previewWrap.style.display = 'block';
          if (previewActions) previewActions.style.display = 'flex';
          dropzone.style.display = 'none';
        }
      } else {
        imageFile = null;
        isNewImageSelected = false;
        dropzone.style.display = 'block';
        previewWrap.style.display = 'none';
        if (previewActions) previewActions.style.display = 'none';
      }

      if (infoBanner && infoText) {
        const nameDisplay = patientName ? `patient ${patientName} (${patientId})` : `patient ${patientId || 'record'}`;
        infoText.textContent = `Predict Again mode for ${nameDisplay}. Patient details and clinical inputs are prefilled. Click "Run AI Assessment" to rerun or select a new image.`;
        infoBanner.style.display = 'block';
      }
    }
  }

  // Load clinical schema AND persisted report concurrently
  const targetReportId = reportIdParam || (prefill && prefill.report_id);
  const schemaPromise = fetch(API_BASE + '/clinical-options').then((r) => {
    if (!r.ok) throw new Error('HTTP ' + r.status);
    return r.json();
  });
  const reportPromise = targetReportId
    ? fetch(`${API_BASE}/reports/${encodeURIComponent(targetReportId)}`).then((r) => (r.ok ? r.json() : null)).catch(() => null)
    : Promise.resolve(null);

  Promise.all([schemaPromise, reportPromise])
    .then(([schemaData, fetchedReport]) => {
      schema = schemaData;
      renderClinicalFields(schemaData, fetchedReport);
      setupWorkflowMode(fetchedReport);
    })
    .catch((err) => {
      console.error(err);
      if (fieldsWrap) {
        fieldsWrap.innerHTML = '<p class="text-muted">Could not load schema from <span class="mono">' +
          API_BASE + '</span>. Ensure backend is running.</p>';
      }
    });

  // Predict submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (submitBtn.disabled) return;
    clearAlert();
    setStatus('');

    let valid = true;
    form.querySelectorAll('.field').forEach((f) => clearFieldError(f));

    form.querySelectorAll('[data-required="true"]').forEach((input) => {
      if (!input.value || !input.value.trim()) {
        setFieldError(fieldOf(input), 'This field is required.');
        valid = false;
      }
    });

    if (mode === 'change_image' && !isNewImageSelected) {
      showAlert('Please select a new skin lesion image to complete image change, or use Predict Again to evaluate with the current image.');
      valid = false;
    }

    if (!imageFile) {
      showAlert('Please upload or select a skin lesion image before submitting.');
      valid = false;
    }
    if (!valid) return;

    const fd = new FormData();
    fd.append('image', imageFile, imageFile.name);

    form.querySelectorAll('[data-feature]').forEach((el) => {
      const v = (el.value || '').trim();
      if (v !== '') fd.append(el.getAttribute('data-feature'), v);
    });

    const origText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Running Independent Inference…';
    setStatus('Executing EfficientNetB0 and Clinical XGBoost models…');

    try {
      const resp = await fetch(API_BASE + '/predict', { method: 'POST', body: fd });
      const data = await resp.json().catch(() => ({}));

      if (!resp.ok) {
        throw new Error(data.detail || 'Inference request failed.');
      }

      // Automatically persist report and patient into SQLite
      const patIdVal = (document.getElementById('predId').value || '').trim() ||
        ('PT-' + Math.floor(10000 + Math.random() * 89999));
      const patNameVal = (document.getElementById('predName').value || '').trim() || 'Patient Record';
      const ageVal = parseInt(document.getElementById('pf_age')?.value || '50', 10);
      const genderVal = document.getElementById('pf_gender')?.value || 'Other';

      const reportPayload = {
        patient_id: patIdVal,
        patient_name: patNameVal,
        patient_age: ageVal,
        patient_gender: genderVal,
        efficientnet_score: data.models.efficientnet_b0.cancer_score,
        clinical_score: data.models.clinical_xgboost.cancer_score,
        clinical_inputs: data.clinical_inputs,
      };

      const saveResp = await fetch(API_BASE + '/reports', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reportPayload),
      });

      const savedReport = saveResp.ok ? await saveResp.json() : null;
      const repId = savedReport ? savedReport.report_id : '';

      sessionStorage.removeItem(STORAGE.PREDICT_PREFILL);
      sessionStorage.setItem(STORAGE.LIVE_IMAGE, previewImg.src);
      sessionStorage.setItem(STORAGE.LIVE_RESULT, JSON.stringify(data));
      sessionStorage.setItem(STORAGE.LIVE_PATIENT, JSON.stringify({ patientId: patIdVal, name: patNameVal, age: ageVal, gender: genderVal }));

      window.location.href = `results.html?report_id=${encodeURIComponent(repId)}`;
    } catch (err) {
      showAlert('Assessment failed: ' + err.message);
      setStatus('');
      submitBtn.disabled = false;
      submitBtn.textContent = origText;
    }
  });
}

/* ---------------------------------------------------------------------- */
/* RESULTS PAGE (results.html)                                            */
/* ---------------------------------------------------------------------- */
async function initResultsPage() {
  const analyzingPanel = document.getElementById('analyzingPanel');
  const resultsContent = document.getElementById('resultsContent');
  if (!analyzingPanel || !resultsContent) return;

  const urlParams = new URLSearchParams(window.location.search);
  const reportId = urlParams.get('report_id');
  const pid = urlParams.get('pid');

  // Case 1: Displaying a persisted report from SQLite via report_id
  if (reportId) {
    try {
      const resp = await fetch(API_BASE + `/reports/${encodeURIComponent(reportId)}`);
      if (!resp.ok) throw new Error('Report not found');
      const rep = await resp.json();
      renderReportRecord(rep);
      analyzingPanel.style.display = 'none';
      resultsContent.style.display = 'block';
      return;
    } catch (e) {
      console.warn('Could not load report by ID:', e);
    }
  }

  // Case 2: Displaying from session cache (live result)
  const cachedResult = sessionStorage.getItem(STORAGE.LIVE_RESULT);
  const cachedImg = sessionStorage.getItem(STORAGE.LIVE_IMAGE);
  const cachedPat = sessionStorage.getItem(STORAGE.LIVE_PATIENT);

  if (cachedResult) {
    try {
      const parsedRes = JSON.parse(cachedResult);
      const parsedPat = cachedPat ? JSON.parse(cachedPat) : {};
      renderLiveEnsemble(parsedRes, cachedImg, parsedPat, reportId);
      analyzingPanel.style.display = 'none';
      resultsContent.style.display = 'block';
      return;
    } catch (e) {}
  }

  // Case 3: Patient ID passed with no active session
  if (pid) {
    try {
      const resp = await fetch(API_BASE + `/patients/${encodeURIComponent(pid)}/reports`);
      if (resp.ok) {
        const reports = await resp.json();
        if (reports.length) {
          renderReportRecord(reports[0]);
          analyzingPanel.style.display = 'none';
          resultsContent.style.display = 'block';
          return;
        }
      }
    } catch (e) {}
  }

  // Empty state fallback
  analyzingPanel.style.display = 'none';
  resultsContent.style.display = 'block';
  resultsContent.innerHTML = `<div class="empty-state" style="text-align:center; padding:50px;">
    <h3>No assessment found</h3>
    <p class="text-muted">Run an AI prediction or select a patient from the archive.</p>
    <a href="predict.html" class="btn btn-primary" style="margin-top:14px;">+ Run AI Prediction</a>
  </div>`;
}

function renderReportRecord(rep) {
  document.getElementById('resPatientName').textContent = rep.patient_name || 'Patient';
  document.getElementById('resPatientMeta').textContent = `${rep.patient_id} · Age ${rep.patient_age || 'N/A'} · ${rep.patient_gender || ''}`;

  const repRef = document.getElementById('reportRefId');
  if (repRef) repRef.textContent = `Report Reference: ${rep.report_id}`;

  const cachedImg = sessionStorage.getItem(STORAGE.LIVE_IMAGE);
  const imgEl = document.getElementById('resLesionImg');
  if (imgEl) imgEl.src = cachedImg || placeholderLesion();

  // Model 1: EfficientNetB0
  const effScore = Number(rep.efficientnet_score || 0);
  const effHigh = effScore >= 0.5;
  document.getElementById('effPredictedClass').textContent = effHigh ? 'Cancer Pattern Suspected' : 'Benign Lesion Pattern';
  document.getElementById('effMalignantPct').textContent = pct(effScore);
  document.getElementById('effBenignPct').textContent = pct(1 - effScore);
  document.getElementById('effMalignantBar').style.width = pct(effScore);
  document.getElementById('effBenignBar').style.width = pct(1 - effScore);

  // Model 2: Clinical XGBoost
  const cliScore = Number(rep.clinical_score || 0);
  const cliHigh = cliScore >= 0.5;
  document.getElementById('cliPredictedClass').textContent = cliHigh ? 'High-Risk Profile' : 'Low-Risk Profile';
  document.getElementById('cliMalignantPct').textContent = pct(cliScore);
  document.getElementById('cliBenignPct').textContent = pct(1 - cliScore);
  document.getElementById('cliMalignantBar').style.width = pct(cliScore);
  document.getElementById('cliBenignBar').style.width = pct(1 - cliScore);

  // Overview / Status Card
  const isLegacy = (rep.resnet50_score != null && rep.ensemble_score != null);
  const pill = document.getElementById('fusionRiskPill');

  if (isLegacy) {
    document.getElementById('fusionPredictedClass').textContent = `${rep.prediction_label || 'Archived'} (Historical 3-Model)`;
    if (pill) {
      pill.textContent = 'Archived 3-Model Record';
      pill.className = `risk-pill ${rep.risk_level || 'low'}`;
    }
    document.getElementById('fusionExplanation').textContent =
      `Historical Archived Record: Evaluated under prior 3-model pipeline with ResNet50. Archived Ensemble Score: ${pct(rep.ensemble_score)}. Note: ResNet50 has since been disconnected from active inference.`;
  } else {
    const summaryTitle = (effHigh && cliHigh)
      ? 'Elevated Risk Indicated Across Both Models'
      : ((effHigh || cliHigh) ? 'Discordant Evaluation (Single Elevated Model)' : 'Low Risk Indicated Across Both Models');
    document.getElementById('fusionPredictedClass').textContent = summaryTitle;
    if (pill) {
      pill.textContent = 'Dual-Model (Uncombined)';
      pill.className = `risk-pill ${effHigh || cliHigh ? 'high' : 'low'}`;
    }
    document.getElementById('fusionExplanation').textContent =
      `Independent Model Outputs: EfficientNetB0 evaluates skin lesion imagery alone (${pct(effScore)}), while Clinical-Only XGBoost evaluates demographic and symptom features alone (${pct(cliScore)}). Because an ensemble combining these two models has not been statistically validated on paired testing data, results are presented as separate model outputs. No combined ensemble score or final merged classification is generated.`;
  }

  // Wire Download PDF button
  const downloadBtn = document.getElementById('generateReportBtn');
  if (downloadBtn) {
    downloadBtn.textContent = 'Download Official PDF Report';
    downloadBtn.onclick = () => {
      window.location.href = `${API_BASE}/reports/${encodeURIComponent(rep.report_id)}/pdf`;
    };
  }

  // Wire Print button
  const printBtn = document.getElementById('printReportBtn');
  if (printBtn) {
    printBtn.onclick = () => window.print();
  }

  // Wire Change Image button
  const changeImgBtn = document.getElementById('changeImageBtn');
  if (changeImgBtn) {
    changeImgBtn.onclick = () => {
      let cliInputs = rep.clinical_inputs || rep.clinical_inputs_json;
      if (typeof cliInputs === 'string') {
        try { cliInputs = JSON.parse(cliInputs); } catch (e) { cliInputs = {}; }
      }
      const cachedImg = sessionStorage.getItem(STORAGE.LIVE_IMAGE) || (imgEl && imgEl.src);
      const isRealImage = cachedImg && cachedImg.startsWith('data:image/') && !cachedImg.includes('svg');
      const prefillPayload = {
        mode: 'change_image',
        report_id: rep.report_id || '',
        patient_id: rep.patient_id || '',
        name: rep.patient_name || '',
        age: rep.patient_age != null ? rep.patient_age : '',
        gender: rep.patient_gender || '',
        clinical_inputs: cliInputs || {},
        keep_image: Boolean(isRealImage),
        image_data_url: isRealImage ? cachedImg : null,
      };
      sessionStorage.setItem(STORAGE.PREDICT_PREFILL, JSON.stringify(prefillPayload));
      window.location.href = `predict.html?mode=change_image&report_id=${encodeURIComponent(rep.report_id || '')}`;
    };
  }

  // Wire Predict Again button
  const predAgainBtn = document.getElementById('predictAgainBtn');
  if (predAgainBtn) {
    predAgainBtn.onclick = () => {
      let cliInputs = rep.clinical_inputs || rep.clinical_inputs_json;
      if (typeof cliInputs === 'string') {
        try { cliInputs = JSON.parse(cliInputs); } catch (e) { cliInputs = {}; }
      }
      const cachedImg = sessionStorage.getItem(STORAGE.LIVE_IMAGE) || (imgEl && imgEl.src);
      const isRealImage = cachedImg && cachedImg.startsWith('data:image/') && !cachedImg.includes('svg');
      const prefillPayload = {
        mode: 'predict_again',
        report_id: rep.report_id || '',
        patient_id: rep.patient_id || '',
        name: rep.patient_name || '',
        age: rep.patient_age != null ? rep.patient_age : '',
        gender: rep.patient_gender || '',
        clinical_inputs: cliInputs || {},
        keep_image: Boolean(isRealImage),
        image_data_url: isRealImage ? cachedImg : null,
      };
      sessionStorage.setItem(STORAGE.PREDICT_PREFILL, JSON.stringify(prefillPayload));
      window.location.href = `predict.html?mode=predict_again&report_id=${encodeURIComponent(rep.report_id || '')}`;
    };
  }
}

function renderLiveEnsemble(data, imgData, patient, reportId) {
  document.getElementById('resPatientName').textContent = patient.name || 'Patient';
  document.getElementById('resPatientMeta').textContent = `${patient.patientId || 'PT-—'} · Independent Dual-Model AI Assessment`;

  const repRef = document.getElementById('reportRefId');
  if (repRef && reportId) repRef.textContent = `Report Reference: ${reportId}`;

  const imgEl = document.getElementById('resLesionImg');
  if (imgEl) imgEl.src = imgData || placeholderLesion();

  const eff = Number(data.models.efficientnet_b0.cancer_score || 0);
  const cli = Number(data.models.clinical_xgboost.cancer_score || 0);
  const effHigh = eff >= 0.5;
  const cliHigh = cli >= 0.5;

  // Model 1: EfficientNetB0
  document.getElementById('effPredictedClass').textContent = effHigh ? 'Cancer Pattern Suspected' : 'Benign Lesion Pattern';
  document.getElementById('effMalignantPct').textContent = pct(eff);
  document.getElementById('effBenignPct').textContent = pct(1 - eff);
  document.getElementById('effMalignantBar').style.width = pct(eff);
  document.getElementById('effBenignBar').style.width = pct(1 - eff);

  // Model 2: Clinical XGBoost
  document.getElementById('cliPredictedClass').textContent = cliHigh ? 'High-Risk Profile' : 'Low-Risk Profile';
  document.getElementById('cliMalignantPct').textContent = pct(cli);
  document.getElementById('cliBenignPct').textContent = pct(1 - cli);
  document.getElementById('cliMalignantBar').style.width = pct(cli);
  document.getElementById('cliBenignBar').style.width = pct(1 - cli);

  // Dual-Model Overview
  const summaryTitle = (effHigh && cliHigh)
    ? 'Elevated Risk Indicated Across Both Models'
    : ((effHigh || cliHigh) ? 'Discordant Evaluation (Single Elevated Model)' : 'Low Risk Indicated Across Both Models');
  document.getElementById('fusionPredictedClass').textContent = summaryTitle;

  const pill = document.getElementById('fusionRiskPill');
  if (pill) {
    pill.textContent = 'Dual-Model (Uncombined)';
    pill.className = `risk-pill ${effHigh || cliHigh ? 'high' : 'low'}`;
  }

  const notice = (data.dual_model_status && data.dual_model_status.notice)
    ? data.dual_model_status.notice
    : `Independent dual-model outputs. EfficientNetB0 image score: ${pct(eff)}, Clinical XGBoost score: ${pct(cli)}. No combined ensemble score is generated pending paired validation.`;
  document.getElementById('fusionExplanation').textContent = notice;

  const downloadBtn = document.getElementById('generateReportBtn');
  if (downloadBtn) {
    downloadBtn.textContent = 'Download Official PDF Report';
    downloadBtn.onclick = () => {
      if (reportId) {
        window.location.href = `${API_BASE}/reports/${encodeURIComponent(reportId)}/pdf`;
      } else {
        toast('Saving report to download PDF…', 'info');
      }
    };
  }

  const printBtn = document.getElementById('printReportBtn');
  if (printBtn) {
    printBtn.onclick = () => window.print();
  }

  // Wire Change Image button
  const changeImgBtn = document.getElementById('changeImageBtn');
  if (changeImgBtn) {
    changeImgBtn.onclick = () => {
      let cliInputs = data.clinical_inputs;
      if (typeof cliInputs === 'string') {
        try { cliInputs = JSON.parse(cliInputs); } catch (e) { cliInputs = {}; }
      }
      const isRealImage = imgData && imgData.startsWith('data:image/') && !imgData.includes('svg');
      const prefillPayload = {
        mode: 'change_image',
        report_id: reportId || '',
        patient_id: patient.patientId || patient.patient_id || '',
        name: patient.name || '',
        age: patient.age != null ? patient.age : '',
        gender: patient.gender || '',
        clinical_inputs: cliInputs || {},
        keep_image: Boolean(isRealImage),
        image_data_url: isRealImage ? imgData : null,
      };
      sessionStorage.setItem(STORAGE.PREDICT_PREFILL, JSON.stringify(prefillPayload));
      const q = reportId ? `&report_id=${encodeURIComponent(reportId)}` : '';
      window.location.href = `predict.html?mode=change_image${q}`;
    };
  }

  // Wire Predict Again button
  const predAgainBtn = document.getElementById('predictAgainBtn');
  if (predAgainBtn) {
    predAgainBtn.onclick = () => {
      let cliInputs = data.clinical_inputs;
      if (typeof cliInputs === 'string') {
        try { cliInputs = JSON.parse(cliInputs); } catch (e) { cliInputs = {}; }
      }
      const isRealImage = imgData && imgData.startsWith('data:image/') && !imgData.includes('svg');
      const prefillPayload = {
        mode: 'predict_again',
        report_id: reportId || '',
        patient_id: patient.patientId || patient.patient_id || '',
        name: patient.name || '',
        age: patient.age != null ? patient.age : '',
        gender: patient.gender || '',
        clinical_inputs: cliInputs || {},
        keep_image: Boolean(isRealImage),
        image_data_url: isRealImage ? imgData : null,
      };
      sessionStorage.setItem(STORAGE.PREDICT_PREFILL, JSON.stringify(prefillPayload));
      const q = reportId ? `&report_id=${encodeURIComponent(reportId)}` : '';
      window.location.href = `predict.html?mode=predict_again${q}`;
    };
  }
}

function placeholderLesion() {
  return 'data:image/svg+xml;utf8,' + encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="300"><rect width="300" height="300" fill="#E2E8ED"/><circle cx="150" cy="150" r="70" fill="#6B4230"/></svg>`
  );
}

/* ---------------------------------------------------------------------- */
/* PROFILE PAGE                                                           */
/* ---------------------------------------------------------------------- */
function initProfilePage() {
  const editBtn = document.getElementById('editProfileBtn');
  if (!editBtn) return;
  const viewMode = document.getElementById('profileView');
  const editMode = document.getElementById('profileEdit');
  const session = getSession();

  if (session) {
    editBtn.addEventListener('click', () => {
      document.getElementById('editName').value = session.fullName;
      document.getElementById('editSpecialization').value = session.specialization;
      document.getElementById('editHospital').value = session.hospital;
      document.getElementById('editPhone').value = session.phone;
      viewMode.style.display = 'none';
      editMode.style.display = 'block';
    });

    document.getElementById('cancelEditBtn')?.addEventListener('click', () => {
      editMode.style.display = 'none';
      viewMode.style.display = 'block';
    });

    document.getElementById('profileEditForm')?.addEventListener('submit', (e) => {
      e.preventDefault();
      const updated = {
        ...session,
        fullName: document.getElementById('editName').value.trim(),
        specialization: document.getElementById('editSpecialization').value.trim(),
        hospital: document.getElementById('editHospital').value.trim(),
        phone: document.getElementById('editPhone').value.trim(),
      };
      setSession(updated, true);
      toast('Profile updated.', 'success');
      setTimeout(() => window.location.reload(), 500);
    });
  }
}

/* ---------------------------------------------------------------------- */
/* Global Lifecycle Initialization                                        */
/* ---------------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', () => {
  initPasswordToggles();
  initLoginPage();
  initSignupPage();
  initAppShell();
  initDashboardPage();
  initPatientsPage();
  initReportsPage();
  initPatientFormPage();
  initPredictPage();
  initResultsPage();
  initProfilePage();
});
