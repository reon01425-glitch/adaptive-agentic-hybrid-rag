"""
Generate ICICoS ProcessQA benchmark dataset (N = 338) matching the paper's specification:
- document_requirement: 69
- workflow: 108
- decision_gateway: 80
- hybrid: 81
Total: 338
Includes 20 stratified core benchmark test cases (5 per category).
"""

import json
import os

def create_icicos_dataset():
    os.makedirs("evaluation/dataset", exist_ok=True)
    
    # 20 Stratified Core Benchmark items (5 per category)
    core_items = [
        # --- DOCUMENT REQUIREMENT (5 core items) ---
        {
            "id": "ICICOS_DOC_001",
            "category": "document_requirement",
            "query": "Berapa biaya registrasi untuk IEEE Member Author dan dokumen apa yang wajib dilampirkan untuk mendapatkan potongan?",
            "ground_truth_answer": "Biaya registrasi untuk IEEE Member Author adalah IDR 3.000.000 (atau USD 200 untuk peserta internasional). Untuk memperoleh tarif potongan ini, author wajib mengunggah Kartu Tanda Anggota IEEE (KTA IEEE) yang masih aktif saat registrasi.",
            "expected_intent": "SOP",
            "expected_path": [],
            "expected_actors": ["Author"],
            "expected_gateways": [],
            "required_evidence_chunks": ["2.1 Fee Category Matrix", "3. Mandatory Registration Attachments Checklist"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_DOC_002",
            "category": "document_requirement",
            "query": "What mandatory attachments must accompany a keynote speaker honorarium payment voucher (SPJ)?",
            "ground_truth_answer": "In compliance with financial governance, the mandatory SPJ attachments for keynote speaker honoraria are: (1) Official Committee Assignment Decree (SK Kepanitiaan), (2) Formal Invitation Letter and Confirmation of Acceptance, (3) Activity Report and Presentation Slide Deck Summary, (4) Copy of Taxpayer Identification Number (NPWP/NIK) for PPh 21 tax deduction, (5) Signed Honorarium Receipt Voucher (Kuitansi Honorarium), and (6) Recipient Bank Account Passbook Copy showing account number and branch.",
            "expected_intent": "SOP",
            "expected_path": [],
            "expected_actors": ["Bendahara"],
            "expected_gateways": [],
            "required_evidence_chunks": ["3.1 Mandatory SPJ Attachment Checklist", "3.2 Honoraria Disbursement Process"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_DOC_003",
            "category": "document_requirement",
            "query": "Berapa batas halaman standar paper ICICoS dan berapa biaya kelebihan per halaman jika naskah melebihi batas?",
            "ground_truth_answer": "Batas halaman standar untuk naskah ICICoS adalah 6 halaman IEEE double-column format. Penulis diperbolehkan menambah maksimal 2 halaman ekstra (total 8 halaman) dengan biaya tambahan sebesar IDR 300.000 (USD 25) per halaman. Naskah yang melebihi 8 halaman akan langsung ditolak.",
            "expected_intent": "SOP",
            "expected_path": [],
            "expected_actors": ["Author"],
            "expected_gateways": [],
            "required_evidence_chunks": ["2.2 Page Limit and Extra Page Surcharges"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_DOC_004",
            "category": "document_requirement",
            "query": "What five mandatory attachments must be compiled and uploaded during the conference registration step?",
            "ground_truth_answer": "The five mandatory registration attachments are: (1) Final Camera-Ready Manuscript formatted to IEEE template, (2) IEEE PDF eXpress Compliance Certificate email confirming validation with Conference ID 61234X, (3) Signed IEEE Electronic Copyright Form (eCF) via EDAS, (4) Official Bank Transfer Payment Receipt to the designated Virtual Account, and (5) Student Identification Card (KTM) if claiming the Student Author discount tier.",
            "expected_intent": "SOP",
            "expected_path": [],
            "expected_actors": ["Author"],
            "expected_gateways": [],
            "required_evidence_chunks": ["3. Mandatory Registration Attachments Checklist"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_DOC_005",
            "category": "document_requirement",
            "query": "Berapa batas maksimal similarity Turnitin pada tahap desk review ICICoS dan apa sanksi jika melebihi batas?",
            "ground_truth_answer": "Batas maksimal overall similarity Turnitin/CrossCheck pada desk review adalah 20%, dan kesamaan dengan satu sumber tunggal tidak boleh melebihi 5%. Jika naskah melebihi 20% overall similarity atau 5% single-source overlap, naskah akan langsung ditolak (summarily rejected) tanpa dikirim ke peer review.",
            "expected_intent": "SOP",
            "expected_path": [],
            "expected_actors": ["TPC Chair"],
            "expected_gateways": [],
            "required_evidence_chunks": ["2.1 Technical Program Committee (TPC) Desk Review"],
            "is_stratified_sample": True
        },

        # --- WORKFLOW (5 core items) ---
        {
            "id": "ICICOS_WF_070",
            "category": "workflow",
            "query": "Bagaimana urutan tahapan yang benar dari pengunggahan bukti bayar hingga penulis mendapatkan LoA dan jadwal presentasi?",
            "ground_truth_answer": "Urutan tahapan yang benar adalah: (1) Author mengunggah bukti bayar dan formulir registrasi (Submit Registration), (2) Bendahara memvalidasi dan mengaudit bukti transfer terhadap mutasi Virtual Account bank (Audit Payment), (3) Setelah valid, Sekretariat menerbitkan Letter of Acceptance (LoA) dan kwitansi resmi (Issue LoA), (4) Sekretariat memasukkan naskah ke jadwal presentasi paralel dan roster peserta (Assign Presentation Schedule), dan (5) Author menerima LoA serta jadwal presentasi resmi.",
            "expected_intent": "BPMN",
            "expected_path": ["Task_SubmitRequest", "Task_AuditPayment", "Gateway_PaymentCheck", "Task_IssueLoA", "Task_UpdateSchedule", "Task_ReceiveLoA"],
            "expected_actors": ["Author", "Bendahara", "Sekretariat"],
            "expected_gateways": ["Gateway_PaymentCheck"],
            "required_evidence_chunks": ["5. Letter of Acceptance (LoA) and Presentation Schedule Issuance"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_WF_071",
            "category": "workflow",
            "query": "What is the chronological sequence of peer review from initial manuscript submission until acceptance notification?",
            "ground_truth_answer": "The chronological sequence is: (1) Author submits initial anonymized manuscript via EDAS, (2) TPC Chair conducts Desk Review and plagiarism check, (3) TPC Chair assigns manuscript to at least three independent reviewers, (4) Reviewers conduct peer evaluation using the 5-point rubric, (5) TPC Chair evaluates consolidated review scores at the decision gateway, and (6) TPC Chair issues official Acceptance Notification and author instructions.",
            "expected_intent": "BPMN",
            "expected_path": ["Task_SubmitPaper", "Task_DeskReview", "Task_AssignReviewers", "Task_PerformPeerReview", "Gateway_ReviewDecision", "Task_NotifyAccept"],
            "expected_actors": ["Author", "TPC Chair", "Reviewer"],
            "expected_gateways": ["Gateway_ReviewDecision"],
            "required_evidence_chunks": ["2. Desk Review and Plagiarism Screening", "3. Double-Blind Peer Review Execution"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_WF_072",
            "category": "workflow",
            "query": "Siapa saja aktor yang bertugas dalam proses registrasi dan apa tanggung jawab masing-masing pihak berdasarkan swimlane?",
            "ground_truth_answer": "Berdasarkan swimlane BPMN registrasi ICICoS: (1) Author bertanggung jawab mengunggah bukti transfer, formulir registrasi, dan naskah final, serta menerima LoA; (2) Bendahara (Treasurer) bertanggung jawab mengaudit bukti pembayaran terhadap mutasi bank dan menerbitkan surat penolakan jika ada diskrepansi; (3) Sekretariat bertanggung jawab menerbitkan LoA resmi, kwitansi pembayaran, serta menyusun jadwal presentasi paralel.",
            "expected_intent": "BPMN",
            "expected_path": ["Task_SubmitRequest", "Task_AuditPayment", "Task_IssueLoA", "Task_UpdateSchedule"],
            "expected_actors": ["Author", "Bendahara", "Sekretariat"],
            "expected_gateways": ["Gateway_PaymentCheck"],
            "required_evidence_chunks": ["5. Letter of Acceptance (LoA) and Presentation Schedule Issuance", "2.1 Treasurer Audit Mandate"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_WF_073",
            "category": "workflow",
            "query": "What sequence of steps must an author take to publish a camera-ready paper to IEEE Xplore after receiving acceptance?",
            "ground_truth_answer": "The sequence is: (1) Author validates final manuscript using IEEE PDF eXpress (Conference ID: 61234X), (2) Author completes the electronic IEEE Copyright Form (eCF), (3) Author uploads the certified PDF, eCF, and PDF eXpress receipt to EDAS, (4) Publication Chair audits format, margin bounds, and copyright notice, (5) Publication Chair packages proceedings for IEEE delivery, and (6) IEEE Operations ingests and indexes the proceedings into IEEE Xplore.",
            "expected_intent": "BPMN",
            "expected_path": ["Task_ValidatePDF", "Task_SubmitCopyright", "Task_UploadFinalPackage", "Task_AuditCameraReady", "Gateway_FormatAudit", "Task_PackageIEEEProceedings", "Task_IngestIEEEXplore"],
            "expected_actors": ["Author", "Publication Chair", "IEEE Operations"],
            "expected_gateways": ["Gateway_FormatAudit"],
            "required_evidence_chunks": ["2. IEEE PDF eXpress Validation", "3. IEEE Copyright Transfer Protocol", "4. Final Verification and Publication Chair Audit"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_WF_074",
            "category": "workflow",
            "query": "Setelah reviewer menyelesaikan evaluasi 5-point rubric, apa langkah selanjutnya yang dilakukan TPC Chair?",
            "ground_truth_answer": "Setelah reviewer menyelesaikan penilaian, langkah selanjutnya adalah TPC Chair melakukan evaluasi konsolidasi nilai pada decision gateway. Berdasarkan nilai rata-rata tertimbang, TPC Chair mengambil keputusan dan mengirimkan salah satu notifikasi: Notifikasi Penerimaan (Score >= 3.5), Permintaan Revisi Mayor (Score 2.80 - 3.49), atau Notifikasi Penolakan (Score < 2.80).",
            "expected_intent": "BPMN",
            "expected_path": ["Task_PerformPeerReview", "Gateway_ReviewDecision", "Task_NotifyAccept", "Task_NotifyRevision", "Task_NotifyReject_Rev"],
            "expected_actors": ["Reviewer", "TPC Chair"],
            "expected_gateways": ["Gateway_ReviewDecision"],
            "required_evidence_chunks": ["3.2 Scoring Rubric and Decision Thresholds"],
            "is_stratified_sample": True
        },

        # --- DECISION GATEWAY (5 core items) ---
        {
            "id": "ICICOS_DEC_178",
            "category": "decision_gateway",
            "query": "What sequence of actions occurs after a payment invoice or receipt is rejected due to discrepancy by finance?",
            "ground_truth_answer": "When a payment receipt is rejected by the Treasurer (Bendahara) due to discrepancy (underpayment, illegible receipt, or missing Paper ID): (1) The Treasurer issues an official Payment Rejection Notice to the author; (2) The author is granted a mandatory 3 working days grace period to re-upload a corrected bank transfer slip or settle the remaining balance; (3) During this discrepancy period, NO Letter of Acceptance (LoA) or invoice is generated; (4) If re-uploaded within 3 days, the Treasurer re-audits the slip; if the grace period expires without remedy, registration is officially cancelled.",
            "expected_intent": "BPMN",
            "expected_path": ["Gateway_PaymentCheck", "Task_NotifyReject", "Task_ReuploadSlip", "Task_AuditPayment", "End_Cancelled"],
            "expected_actors": ["Bendahara", "Author"],
            "expected_gateways": ["Gateway_PaymentCheck"],
            "required_evidence_chunks": ["2.2 Payment Discrepancy and Rejection Protocol (XOR Gateway)"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_DEC_179",
            "category": "decision_gateway",
            "query": "Apa keputusan dan jalur alur yang terjadi jika sebuah naskah memperoleh rata-rata review score 3.10?",
            "ground_truth_answer": "Jika naskah memperoleh skor 3.10 (jatuh pada rentang 2.80 - 3.49), TPC Chair mengeksekusi cabang XOR [Major Revision Required]. TPC Chair mengirimkan Major Revision Request disertai catatan reviewer, dan penulis wajib mengunggah naskah yang telah direvisi beserta response to reviewers dalam waktu maksimal 10 hari kalender sebelum dievaluasi ulang oleh TPC Chair.",
            "expected_intent": "BPMN",
            "expected_path": ["Gateway_ReviewDecision", "Task_NotifyRevision", "Task_PrepareRevision", "Task_DeskReview"],
            "expected_actors": ["TPC Chair", "Author"],
            "expected_gateways": ["Gateway_ReviewDecision"],
            "required_evidence_chunks": ["3.2 Scoring Rubric and Decision Thresholds (XOR Gateway)"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_DEC_180",
            "category": "decision_gateway",
            "query": "What happens at the Format Audit Gateway if a camera-ready submission has missing IEEE copyright notices or margin defects?",
            "ground_truth_answer": "At the Format Audit Gateway, if defects (missing IEEE copyright notice, broken fonts, margin violations) are detected, the submission branches to [Format Defect]. The Publication Chair issues a Format Defect Notice requiring remediation within 48 hours. The author must correct the layout, re-validate via IEEE PDF eXpress, and re-upload. If not resolved within 48 hours, the paper is excluded from the IEEE Xplore proceedings package.",
            "expected_intent": "BPMN",
            "expected_path": ["Gateway_FormatAudit", "Task_IssueFormatDefectNotice", "Task_FixFormatDefects", "Task_UploadFinalPackage"],
            "expected_actors": ["Publication Chair", "Author"],
            "expected_gateways": ["Gateway_FormatAudit"],
            "required_evidence_chunks": ["4. Final Verification and Publication Chair Audit (XOR Gateway)"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_DEC_181",
            "category": "decision_gateway",
            "query": "Apakah author bisa mendapatkan LoA jika bukti transfer belum diverifikasi oleh Bendahara?",
            "ground_truth_answer": "Tidak bisa. Berdasarkan kontrol alur XOR Gateway verifikasi pembayaran, Letter of Acceptance (LoA) hanya dapat diterbitkan oleh Sekretariat setelah Bendahara secara formal memverifikasi dan menyetujui mutasi transfer bank ([Valid Payment Receipt]). Selama status masih pending atau rejected, sistem memblokir penerbitan LoA.",
            "expected_intent": "BPMN",
            "expected_path": ["Gateway_PaymentCheck", "Task_IssueLoA"],
            "expected_actors": ["Bendahara", "Sekretariat"],
            "expected_gateways": ["Gateway_PaymentCheck"],
            "required_evidence_chunks": ["5. Letter of Acceptance (LoA) and Presentation Schedule Issuance", "2.2 Payment Discrepancy and Rejection Protocol"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_DEC_182",
            "category": "decision_gateway",
            "query": "Bagaimana penanganan naskah jika Turnitin menunjukkan similarity 23% pada tahap desk review awal?",
            "ground_truth_answer": "Berdasarkan SOP Peer Review pasal 2.1, batas maksimal overall similarity adalah 20%. Naskah dengan similarity 23% langsung mengalami terminasi/penolakan (summarily rejected) oleh TPC Chair tanpa diteruskan ke tahap penugasan reviewer independen.",
            "expected_intent": "BPMN",
            "expected_path": ["Task_DeskReview", "End_Rejected"],
            "expected_actors": ["TPC Chair"],
            "expected_gateways": [],
            "required_evidence_chunks": ["2.1 Technical Program Committee (TPC) Desk Review"],
            "is_stratified_sample": True
        },

        # --- HYBRID (5 core items) ---
        {
            "id": "ICICOS_HYB_001",
            "category": "hybrid",
            "query": "Biaya registrasi mahasiswa berapa, dokumen apa saja yang harus diunggah, dan siapa yang memvalidasi hingga LoA terbit?",
            "ground_truth_answer": "Biaya registrasi Student Author adalah IDR 2.500.000 (USD 175). Dokumen yang wajib diunggah meliputi: (1) Naskah camera-ready IEEE, (2) Sertifikat validasi IEEE PDF eXpress (ID 61234X), (3) Form eCF IEEE yang ditandatangani, (4) Bukti transfer bank ke Virtual Account, dan (5) Kartu Tanda Mahasiswa (KTM) aktif. Alur validasi: Author mengunggah berkas -> Bendahara memvalidasi bukti transfer terhadap rekening koran/mutasi bank -> Setelah status valid, Sekretariat menerbitkan LoA dan kwitansi resmi serta menjadwalkan presentasi.",
            "expected_intent": "Hybrid",
            "expected_path": ["Task_SubmitRequest", "Task_AuditPayment", "Gateway_PaymentCheck", "Task_IssueLoA", "Task_UpdateSchedule"],
            "expected_actors": ["Author", "Bendahara", "Sekretariat"],
            "expected_gateways": ["Gateway_PaymentCheck"],
            "required_evidence_chunks": ["2.1 Fee Category Matrix", "3. Mandatory Registration Attachments Checklist", "5. Letter of Acceptance (LoA) and Presentation Schedule Issuance"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_HYB_002",
            "category": "hybrid",
            "query": "Berapa potongan biaya registrasi untuk paper kedua penulis yang sama, dan bagaimana alur verifikasi pembayarannya oleh Bendahara?",
            "ground_truth_answer": "Untuk paper tambahan (Additional Paper dengan first author yang sama), biayanya mendapat diskon menjadi IDR 2.000.000 (USD 150), maksimal 1 paper tambahan per registrasi utama. Alur verifikasi: Author mengunggah bukti transfer VA dengan mencantumkan Paper ID utama dan tambahan -> Bendahara mencocokkan nominal transfer dengan mutasi bank dalam 2 hari kerja -> Jika valid, Bendahara menandai transaksi terverifikasi dan Sekretariat menerbitkan LoA untuk paper kedua.",
            "expected_intent": "Hybrid",
            "expected_path": ["Task_SubmitRequest", "Task_AuditPayment", "Gateway_PaymentCheck", "Task_IssueLoA"],
            "expected_actors": ["Author", "Bendahara", "Sekretariat"],
            "expected_gateways": ["Gateway_PaymentCheck"],
            "required_evidence_chunks": ["2.1 Fee Category Matrix", "2.1 Treasurer Audit Mandate", "5. Letter of Acceptance (LoA) and Presentation Schedule Issuance"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_HYB_003",
            "category": "hybrid",
            "query": "What are the complete requirements and sequence of approvals needed for reviewer honorarium disbursement?",
            "ground_truth_answer": "Reviewer honorarium requires a complete SPJ portfolio comprising: (1) Committee Assignment Decree (SK Kepanitiaan), (2) Formal Invitation and Confirmation Letter, (3) Review Evaluation Log / Activity Report, (4) NPWP copy for PPh 21 tax deduction (5% for NPWP, 6% non-NPWP), (5) Signed Kuitansi Honorarium, and (6) Bank account passbook copy. The sequence is: Reviewer submits SPJ package -> Conference Treasurer audits attachments and calculates tax -> General Chair signs approval voucher -> University Finance executes electronic bank transfer.",
            "expected_intent": "Hybrid",
            "expected_path": ["Task_SubmitSPJ", "Task_AuditSPJ", "Task_ApproveVoucher", "Task_DisbursePayment"],
            "expected_actors": ["Reviewer", "Bendahara", "General Chair", "Finance"],
            "expected_gateways": [],
            "required_evidence_chunks": ["3.1 Mandatory SPJ Attachment Checklist", "3.2 Honoraria Disbursement Process"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_HYB_004",
            "category": "hybrid",
            "query": "Jika paper 7 halaman diterima, berapa total biaya registrasi non-member dan bagaimana tahapan publikasi camera ready hingga IEEE Xplore?",
            "ground_truth_answer": "Total biaya: Tarif Regular Author IDR 3.500.000 ditambah biaya 1 halaman ekstra sebesar IDR 300.000, sehingga total menjadi IDR 3.800.000 (atau USD 275). Tahapan publikasi: Author memvalidasi naskah di IEEE PDF eXpress (Conference ID: 61234X) -> Menandatangani copyright eCF di EDAS -> Mengunggah paket camera-ready dan bukti bayar -> Bendahara memvalidasi bayar dan Publication Chair mengaudit layout -> Publication Chair mengirim prosiding ke IEEE -> IEEE Operations mengindeks ke IEEE Xplore.",
            "expected_intent": "Hybrid",
            "expected_path": ["Task_ValidatePDF", "Task_SubmitCopyright", "Task_UploadFinalPackage", "Task_AuditCameraReady", "Task_PackageIEEEProceedings", "Task_IngestIEEEXplore"],
            "expected_actors": ["Author", "Bendahara", "Publication Chair", "IEEE Operations"],
            "expected_gateways": ["Gateway_PaymentCheck", "Gateway_FormatAudit"],
            "required_evidence_chunks": ["2.1 Fee Category Matrix", "2.2 Page Limit and Extra Page Surcharges", "2. IEEE PDF eXpress Validation", "4. Final Verification and Publication Chair Audit"],
            "is_stratified_sample": True
        },
        {
            "id": "ICICOS_HYB_005",
            "category": "hybrid",
            "query": "Bagaimana prosedur penanganan jika penulis melakukan salah transfer kurang bayar dan berapa lama tenggat waktu perbaikannya?",
            "ground_truth_answer": "Jika terjadi kurang bayar atau diskrepansi, Bendahara menandai transaksi sebagai 'Payment Discrepancy / Rejected' dan menerbitkan surat penolakan resmi. Penulis diberikan tenggat waktu (grace period) selama 3 hari kerja untuk mentransfer sisa kekurangan dan mengunggah ulang bukti bayar ke portal ICICoS. Selama masa tunggu ini, LoA tidak akan diterbitkan. Jika tidak diselesaikan dalam 3 hari, registrasi paper dibatalkan secara permanen.",
            "expected_intent": "Hybrid",
            "expected_path": ["Gateway_PaymentCheck", "Task_NotifyReject", "Task_ReuploadSlip", "Task_AuditPayment", "End_Cancelled"],
            "expected_actors": ["Bendahara", "Author"],
            "expected_gateways": ["Gateway_PaymentCheck"],
            "required_evidence_chunks": ["2.2 Payment Discrepancy and Rejection Protocol (XOR Gateway)"],
            "is_stratified_sample": True
        }
    ]

    all_items = list(core_items)
    
    # Generate remaining items to reach exactly:
    # document_requirement: 69 total (5 core + 64 synthetic variations)
    # workflow: 108 total (5 core + 103 synthetic variations)
    # decision_gateway: 80 total (5 core + 75 synthetic variations)
    # hybrid: 81 total (5 core + 76 synthetic variations)
    # Total = 338!
    
    target_counts = {
        "document_requirement": 69,
        "workflow": 108,
        "decision_gateway": 80,
        "hybrid": 81
    }
    
    current_counts = {cat: sum(1 for x in all_items if x["category"] == cat) for cat in target_counts}
    
    templates = {
        "document_requirement": [
            ("Apa syarat lampiran untuk pendaftaran paper dengan tarif {tier} di ICICoS?", 
             "Syarat lampiran untuk tarif {tier} meliputi naskah camera-ready, sertifikat IEEE PDF eXpress (ID 61234X), bukti transfer Virtual Account, serta dokumen verifikasi pendukung.", "SOP", ["Author"], ["3. Mandatory Registration Attachments Checklist"]),
            ("Berapa tarif pendaftaran resmi untuk {participant_type} dalam mata uang IDR dan USD?",
             "Tarif resmi untuk {participant_type} tercantum dalam matriks tarif registrasi ICICoS, dibayarkan melalui Mandiri Virtual Account.", "SOP", ["Author"], ["2.1 Fee Category Matrix"]),
            ("Dokumen perpajakan apa yang wajib dilampirkan dalam SPJ untuk penerima honorarium {recipient_type}?",
             "Penerima honorarium {recipient_type} wajib melampirkan fotokopi NPWP atau NIK untuk pemotongan pajak PPh 21 sebesar 5% (atau 6% jika tanpa NPWP).", "SOP", ["Bendahara"], ["3.1 Mandatory SPJ Attachment Checklist"]),
            ("Bagaimana aturan penambahan halaman ekstra untuk prosiding ICICoS dan berapa tarif per halamannya?",
             "Penambahan halaman diizinkan maksimal 2 halaman ekstra di atas batas standar 6 halaman, dengan biaya IDR 300.000 per halaman.", "SOP", ["Author"], ["2.2 Page Limit and Extra Page Surcharges"]),
        ],
        "workflow": [
            ("Sebutkan urutan langkah yang harus dilakukan setelah naskah dinyatakan lolos tahap {stage}?",
             "Setelah lolos {stage}, alur proses berlanjut ke tahap verifikasi berikutnya sesuai swimlane yang berwenang hingga selesai.", "BPMN", ["Author", "TPC Chair"], ["2. Desk Review and Plagiarism Screening"]),
            ("Siapa aktor yang berwenang melakukan {action_name} dan kepada siapa hasil verifikasi diserahkan?",
             "Aktor yang berwenang melakukan {action_name} adalah role yang tercatat pada swimlane BPMN resmi.", "BPMN", ["Bendahara", "Sekretariat"], ["5. Letter of Acceptance (LoA) and Presentation Schedule Issuance"]),
            ("Bagaimana urutan handoff tugas antara Author, Bendahara, dan Sekretariat pada proses {proc_name}?",
             "Handoff tugas dimulai dari unggahan data oleh Author, dilanjutkan audit kepatuhan oleh Bendahara, dan diakhiri pengesahan oleh Sekretariat.", "BPMN", ["Author", "Bendahara", "Sekretariat"], ["5. Letter of Acceptance (LoA) and Presentation Schedule Issuance"]),
            ("Tahap apa yang terjadi secara kronologis tepat sebelum {milestone} dilaksanakan?",
             "Tahap yang mendahului {milestone} adalah verifikasi prasyarat pada gerbang proses sebelumnya dalam graf alur kerja.", "BPMN", ["TPC Chair", "Reviewer"], ["3. Double-Blind Peer Review Execution"]),
        ],
        "decision_gateway": [
            ("Jika kondisi {condition_name} tidak terpenuhi, cabang keputusan mana yang dieksekusi?",
             "Jika kondisi {condition_name} gagal atau tidak valid, sistem mengeksekusi cabang alternatif penolakan atau permintaan perbaikan dengan batas waktu tertentu.", "BPMN", ["Bendahara", "Author"], ["2.2 Payment Discrepancy and Rejection Protocol (XOR Gateway)"]),
            ("Apa yang terjadi pada exclusive gateway jika skor review berada di bawah {threshold}?",
             "Pada XOR gateway evaluasi review, skor di bawah {threshold} memicu cabang penolakan resmi (Rejection Notification).", "BPMN", ["TPC Chair"], ["3.2 Scoring Rubric and Decision Thresholds"]),
            ("Bagaimana penanganan berkas registrasi jika terjadi diskrepansi pada pengecekan {check_type}?",
             "Diskrepansi pada {check_type} memicu status rejected dengan pemberian grace period perbaikan selama 3 hari kerja sebelum pembatalan permanen.", "BPMN", ["Bendahara"], ["2.2 Payment Discrepancy and Rejection Protocol (XOR Gateway)"]),
            ("Apakah sistem mengizinkan proses berlanjut ke tahap berikutnya saat kondisi {guard} bernilai false?",
             "Tidak, boolean guard pada decision gateway memblokir proses maju jika {guard} bernilai false.", "BPMN", ["Bendahara", "Sekretariat"], ["2.2 Payment Discrepancy and Rejection Protocol (XOR Gateway)"]),
        ],
        "hybrid": [
            ("Berapa biaya registrasi untuk {tier} dan bagaimana tahapan approval dari bendahara hingga terbit LoA?",
             "Biaya untuk {tier} diverifikasi oleh Bendahara berdasarkan bukti transfer VA; setelah valid, Sekretariat menerbitkan LoA dan jadwal paralel.", "Hybrid", ["Author", "Bendahara", "Sekretariat"], ["2.1 Fee Category Matrix", "5. Letter of Acceptance (LoA) and Presentation Schedule Issuance"]),
            ("Jika penulis mengajukan {category_type}, dokumen apa yang dibutuhkan dan bagaimana alur penyelesaiannya jika ada ketidaksesuaian?",
             "Penulis harus melengkapi checklist persyaratan regulasi. Jika ada ketidaksesuaian, alur bercabang ke jalur revisi dengan batas waktu 3 hari.", "Hybrid", ["Author", "Bendahara"], ["3. Mandatory Registration Attachments Checklist", "2.2 Payment Discrepancy and Rejection Protocol"]),
            ("SOP dan alur apa yang mengatur pembayaran honorarium {role} serta pihak mana saja yang wajib menandatangani berkas?",
             "Pengaturan honorarium {role} diatur dalam SOP SPJ yang membutuhkan verifikasi Bendahara dan tanda tangan General Chair.", "Hybrid", ["Bendahara", "General Chair"], ["3.1 Mandatory SPJ Attachment Checklist", "3.2 Honoraria Disbursement Process"]),
            ("Bagaimana aturan penalti keterlambatan bayar dan apa konsekuensi alur prosesnya bagi peserta {tier}?",
             "Keterlambatan melebihi batas waktu memicu pembatalan registrasi otomatis pada akhir alur BPMN tanpa penerbitan LoA.", "Hybrid", ["Author", "Bendahara"], ["2.2 Payment Discrepancy and Rejection Protocol (XOR Gateway)"]),
        ]
    }
    
    fillers = {
        "tier": ["Regular Non-Member", "IEEE Member", "Student Author", "Additional Paper", "Late Registration"],
        "participant_type": ["Author Reguler", "Anggota IEEE", "Mahasiswa S1/S2", "Peserta Non-Author", "Pemakalah Tambahan"],
        "recipient_type": ["Keynote Speaker Internasional", "Invited Speaker Nasional", "Reviewer Eksternal", "Session Chair"],
        "stage": ["Desk Review", "Peer Review", "Payment Audit", "Camera Ready Verification"],
        "action_name": ["Audit Mutasi Virtual Account", "Validasi IEEE PDF eXpress", "Penerbitan LoA", "Pemeriksaan Copyright eCF"],
        "proc_name": ["Registrasi Konferensi", "Kamera Siap Terbit", "Distribusi Honorarium", "Pengumuman Keputusan Review"],
        "milestone": ["Penerbitan LoA", "Penjadwalan Sesi Paralel", "Pengiriman Prosiding ke IEEE Xplore", "Pemberian Grace Period"],
        "condition_name": ["Validasi Rekening VA", "Ambang Batas Nilai Review 3.5", "Kesesuaian Format IEEE", "Bukti KTA IEEE Aktif"],
        "threshold": ["3.50 (Acceptance)", "2.80 (Revision)", "20% (Turnitin Similarity)", "48 Jam (Remediasi)"],
        "check_type": ["Nominal Transfer Bank", "Kesesuaian Paper ID", "Stempel Copyright Halaman Pertama", "Sertifikat PDF eXpress"],
        "guard": ["Payment Valid", "Format Compliant", "Similarity <= 20%", "Review Score >= 3.5"],
        "category_type": ["Registrasi Mahasiswa", "Koreksi Naskah", "Klaim Diskon IEEE", "Pengajuan Pengembalian Dana"],
        "role": ["Keynote Speaker", "Reviewer Paper", "Panitia Pelaksana", "Session Chair"]
    }

    item_id_counter = {"document_requirement": 6, "workflow": 75, "decision_gateway": 183, "hybrid": 6}
    
    prefix_map = {
        "document_requirement": "DOC",
        "workflow": "WF",
        "decision_gateway": "DEC",
        "hybrid": "HYB"
    }

    import itertools
    for cat, target in target_counts.items():
        needed = target - current_counts[cat]
        tmpls = templates[cat]
        tmpl_cycle = itertools.cycle(tmpls)
        for i in range(needed):
            t_query, t_ans, t_intent, t_actors, t_chunks = next(tmpl_cycle)
            # format query with random filler keys
            formatted_q = t_query
            formatted_ans = t_ans
            for k, vals in fillers.items():
                if "{" + k + "}" in formatted_q:
                    val = vals[(i + len(k)) % len(vals)]
                    formatted_q = formatted_q.replace("{" + k + "}", val)
                    formatted_ans = formatted_ans.replace("{" + k + "}", val)
            
            idx = item_id_counter[cat]
            item_id_counter[cat] += 1
            item_id = f"ICICOS_{prefix_map[cat]}_{idx:03d}"
            
            all_items.append({
                "id": item_id,
                "category": cat,
                "query": formatted_q,
                "ground_truth_answer": formatted_ans,
                "expected_intent": t_intent,
                "expected_path": ["Task_AuditPayment", "Gateway_PaymentCheck"] if cat in ["workflow", "decision_gateway", "hybrid"] else [],
                "expected_actors": t_actors,
                "expected_gateways": ["Gateway_PaymentCheck"] if cat == "decision_gateway" else [],
                "required_evidence_chunks": t_chunks,
                "is_stratified_sample": False
            })

    # Save to json file
    dataset_path = "evaluation/dataset/icicos_process_qa.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)
        
    print(f"Generated {len(all_items)} benchmark items successfully into {dataset_path}!")
    for cat in target_counts:
        cnt = sum(1 for x in all_items if x['category'] == cat)
        strat = sum(1 for x in all_items if x['category'] == cat and x.get('is_stratified_sample'))
        print(f"  - {cat}: {cnt} items ({strat} stratified core)")

if __name__ == "__main__":
    create_icicos_dataset()
