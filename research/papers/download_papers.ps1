# IntelliCredit research pack - open-access paper downloader (SCOUT, 2026-11-06)
# Run from any folder:  powershell -ExecutionPolicy Bypass -File .\download_papers.ps1
# Downloads ONLY open-access copies (arXiv, NBER/FDIC/WB/IMF/ECB working papers,
# proceedings, regulator and government PDFs) into this folder.
# Anything that fails, redirects to a login, or shows a bot check is skipped;
# do not try to bypass it.
# Numbers in brackets match research/REFERENCES_ACM.md.

$ErrorActionPreference = 'Continue'
$dest = Split-Path -Parent $MyInvocation.MyCommand.Path
$ua = 'IntelliCredit-research-pack/1.0 (student capstone; open-access papers only)'

$papers = @(
  @{ f='2020_Berg_digital_footprints_FDIC_WP.pdf';            u='https://www.fdic.gov/system/files/2024-07/cfr-wp2018-04.pdf' }                       # [1]
  @{ f='2020_Bjorkegren_Grissen_mobile_phone_repayment.pdf';  u='https://arxiv.org/pdf/1712.05840' }                                                   # [2]
  @{ f='2019_Oskarsdottir_big_data_credit_scoring.pdf';       u='https://arxiv.org/pdf/2002.09931' }                                                   # [3]
  @{ f='2020_Agarwal_Alok_Ghosh_Gupta_alt_credit_India.pdf';  u='https://www.jbs.cam.ac.uk/wp-content/uploads/2020/08/2020-06-conference-paper-agarwal-alok-ghosh-gupta.pdf' } # [4]
  @{ f='2024_Alok_Ghosh_Kulkarni_Puri_open_banking_UPI.pdf';  u='https://conference.nber.org/conf_papers/f210448/f210448.pdf' }                        # [5] conference version
  @{ f='2015_Lessmann_benchmarking_credit_scoring.pdf';       u='https://crc.business-school.ed.ac.uk/sites/crc/files/2023-10/Benchmarking-State-of-the-Art-Classification-Algorithms-for-Credit-Scoring-Lessmann-Seow-Baesens-and-Thomas.pdf' } # [8]
  @{ f='2024_FCA_synthetic_data_financial_services.pdf';      u='https://www.fca.org.uk/publication/corporate/report-using-synthetic-data-in-financial-services.pdf' }   # [14]
  @{ f='2025_FCA_synthetic_data_governance.pdf';              u='https://www.fca.org.uk/publication/corporate/synthetic-data-models-financial-services-governance.pdf' } # [15]
  @{ f='2017_Esteban_RCGAN_TSTR.pdf';                         u='https://arxiv.org/pdf/1706.02633' }                                                   # [16]
  @{ f='2019_Xu_CTGAN.pdf';                                   u='https://arxiv.org/pdf/1907.00503' }                                                   # [18]
  @{ f='2022_Jordon_synthetic_data_Royal_Society.pdf';        u='https://arxiv.org/pdf/2205.03257' }                                                   # [19]
  @{ f='2023_India_DPDP_Act.pdf';                             u='https://www.indiacode.nic.in/bitstream/123456789/22037/2/a2023-22.pdf' }              # [20]
  @{ f='2005_India_CICRA.pdf';                                u='https://prsindia.org/files/bills_acts/acts_parliament/2005/the-credit-information-companies-(regulation)-act-2005.pdf' } # [22]
  @{ f='2013_Beck_Jakubik_Piloiu_ECB_WP1515_NPL.pdf';         u='https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1515.pdf' }                             # [23] WP version
  @{ f='2011_Nkusu_IMF_NPL_macrofinancial.pdf';               u='https://www.imf.org/external/pubs/ft/wp/2011/wp11161.pdf' }                           # [24]
  @{ f='2021_Angelopoulos_Bates_conformal_intro.pdf';         u='https://arxiv.org/pdf/2107.07511' }                                                   # [32]
  @{ f='2021_Gibbs_Candes_adaptive_conformal.pdf';            u='https://arxiv.org/pdf/2106.00170' }                                                   # [33]
  @{ f='2021_Xu_Xie_EnbPI_conformal_time_series.pdf';         u='https://arxiv.org/pdf/2010.09107' }                                                   # [34]
  @{ f='2021_Stankeviciute_conformal_ts_forecasting.pdf';     u='https://papers.neurips.cc/paper_files/paper/2021/file/312f1ba2a72318edaaa995a67835fad5-Paper.pdf' } # [35]
  @{ f='1994_Agrawal_Srikant_Apriori.pdf';                    u='https://vldb.org/conf/1994/P487.PDF' }                                                # [38]
  @{ f='2017_Lundberg_Lee_SHAP.pdf';                          u='https://arxiv.org/pdf/1705.07874' }                                                   # [43]
  @{ f='2017_Wachter_counterfactual_explanations.pdf';        u='https://arxiv.org/pdf/1711.00399' }                                                   # [44]
  @{ f='2019_Ustun_actionable_recourse.pdf';                  u='https://arxiv.org/pdf/1809.06514' }                                                   # [45]
  @{ f='2020_Mothilal_DiCE_counterfactuals.pdf';              u='https://arxiv.org/pdf/1905.07697' }                                                   # [46]
  @{ f='2020_Karimi_algorithmic_recourse_survey.pdf';         u='https://arxiv.org/pdf/2010.04050' }                                                   # [47]
  @{ f='2019_Bartlett_fintech_lending_discrimination_NBER.pdf'; u='https://www.nber.org/papers/w25943.pdf' }                                           # [50] WP version
  @{ f='2022_Grootendorst_BERTopic.pdf';                      u='https://arxiv.org/pdf/2203.05794' }                                                   # [54]
  @{ f='2018_Bastani_LDA_CFPB_complaints.pdf';                u='https://arxiv.org/pdf/1807.07468' }                                                   # [55] preprint
  @{ f='2024_Vaishnav_CFPB_complaints_ML.pdf';                u='https://arxiv.org/pdf/2407.06399' }                                                   # [56]
  @{ f='2015_CFPB_complaint_narratives_policy.pdf';           u='https://files.consumerfinance.gov/f/201506_cfpb_publication-of-consumer-response-complaint-narratives.pdf' } # [58]
  @{ f='2022_NITI_Aayog_gig_economy_policy_brief.pdf';        u="https://www.niti.gov.in/sites/default/files/2022-06/Policy_Brief_India's_Booming_Gig_and_Platform_Economy_27062022.pdf" } # [64]
  @{ f='2026_MFIN_Micrometer_Q1FY27_press_release.pdf';       u='https://mfinindia.org/assets/upload_image/news/pdf/Micrometer%20Q1%20FY%2026-27%20Press%20Release.pdf' } # [67]
)

$ok = 0; $fail = @()
foreach ($p in $papers) {
  $out = Join-Path $dest $p.f
  if (Test-Path $out) { Write-Host "skip (exists) $($p.f)"; $ok++; continue }
  try {
    Invoke-WebRequest -Uri $p.u -OutFile $out -UserAgent $ua -MaximumRedirection 5 -TimeoutSec 60 -UseBasicParsing
    $bytes = [System.IO.File]::ReadAllBytes($out)
    $head = [System.Text.Encoding]::ASCII.GetString($bytes, 0, [Math]::Min(5, $bytes.Length))
    if ($head -ne '%PDF-') { Remove-Item $out -Force; throw 'not a PDF (login page or bot check?)' }
    Write-Host "ok   $($p.f)"; $ok++
  } catch {
    Write-Host "FAIL $($p.f) : $($_.Exception.Message)"; $fail += $p.f
  }
  Start-Sleep -Milliseconds 800
}
Write-Host ""
Write-Host "Downloaded/present: $ok of $($papers.Count)"
if ($fail.Count) { Write-Host "Failed (open manually in a browser, skip if paywalled):"; $fail | ForEach-Object { Write-Host "  $_" } }
