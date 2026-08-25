# RiVeR OOM Complete Parameter Table

This report flattens the verified JSON outputs into human-readable tables.
The corresponding machine-readable table is `data/final_oom_all_parameters.tsv`.
Long selector vectors are kept in the TSV as `|`-separated entries.

## Overview

| N | oom_kb | repeat_bound | n | ell | p_bits | log2_q | hat_q_bits | hat_n | hat_k | phi_a | phi_s | phi_m | phi_b | tau_g0 | tau_g1 | outer_mlwr_delta | outer_hiding_mlwe_delta | selector_hiding_mlwe_delta | selector_binding_asis_delta | all_checks_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 20.133209060563 | 8.34428179357 | 44 | 54 | 44 | 49.930737337519 | 44 | 42 | 46 | 32 | 26 | 32 | 2 | 3.14 | 2.68 | 1.004647519883 | 1.004469578413 | 1.00468705157 | 1.004667190992 | yes |
| 16 | 21.409119640955 | 8.435178948968 | 41 | 59 | 48 | 53.930737337563 | 46 | 43 | 49 | 40 | 22 | 32 | 2 | 3.09 | 3.08 | 1.004618362335 | 1.004364890621 | 1.004608734875 | 1.00467709742 | yes |
| 64 | 25.535994066824 | 8.625685740071 | 44 | 54 | 44 | 49.930737337519 | 48 | 50 | 51 | 34 | 24 | 32 | 2 | 3.05 | 3.33 | 1.004647519883 | 1.004469578413 | 1.004657331931 | 1.004628035429 | yes |
| 128 | 28.952209111905 | 8.598617463525 | 45 | 54 | 44 | 49.930737337519 | 48 | 49 | 51 | 24 | 34 | 32 | 2 | 3.09 | 3.58 | 1.004647519883 | 1.004487582246 | 1.004657331931 | 1.00468705157 | yes |
| 256 | 36.040998993483 | 8.526235986558 | 42 | 59 | 48 | 53.930737337563 | 49 | 48 | 52 | 22 | 40 | 32 | 2 | 3.06 | 3.84 | 1.004618362335 | 1.004381952541 | 1.004667190992 | 1.004657331931 | yes |

## Moduli And Selector Limits

| N | p | q | hat_q | p_mod_8 | q0_mod_8 | hat_q_mod_8 | hat_q_lower_bound | beta_sel_inf |
|---|---|---|---|---|---|---|---|---|
| 8 | 17592186043877 | 1073123348676497 | 8796093022237 | 5 | 5 | 5 | 4838131712 | 8.055755192839e+12 |
| 16 | 281474976710597 | 17169973579346417 | 35184372088997 | 5 | 5 | 5 | 7557613568 | 2.654289788928e+13 |
| 64 | 17592186043877 | 1073123348676497 | 140737488355333 | 5 | 5 | 5 | 5461379072 | 7.950177738424e+13 |
| 128 | 17592186043877 | 1073123348676497 | 140737488355333 | 5 | 5 | 5 | 2722629632 | 8.090275276653e+13 |
| 256 | 281474976710597 | 17169973579346417 | 281474976710677 | 5 | 5 | 5 | 2288125952 | 1.351716402364e+14 |

## Main Bounds

| N | B_s | eta_m | B_response | sigma_s | sigma_m | beta_sis_1 | beta_sis | beta_sis_2 | beta_sis_2_q_requirement | B_g_0 | B_g_1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 860160 | 86889.281272202963 | 583259.46956050361 | 22364160 | 2780457.000710494816 | 3005743104 | 6.155761876992e+12 | 3005980135.382326602936 | 3005980135.382326602936 | 3933474215.253333568573 | 719407022.080000042915 |
| 16 | 868892.81272202963 | 86889.281272202963 | 563546.191782004666 | 19115641.879884652793 | 2780457.000710494816 | 2595225600 | 5.315022028800e+12 | 2595500121.742427825928 | 2595500121.742427825928 | 12960399360 | 1291845632 |
| 64 | 860160 | 86889.281272202963 | 583259.46956050361 | 20643840 | 2780457.000710494816 | 2774532096 | 5.682241732608e+12 | 2774788878.239883422852 | 2774788878.239883422852 | 38819227238.400001525879 | 1009118085.120000004768 |
| 128 | 864537.43285065447 | 86889.281272202963 | 589695.9861080962 | 29394272.71692225337 | 2780457.000710494816 | 3970695168 | 8.131983704064e+12 | 3970874599.411085128784 | 3970874599.411085128784 | 39503297249.279998779297 | 540561899.519999980927 |
| 256 | 873226.469594228431 | 86889.281272202963 | 570205.276608345681 | 34929058.783769138157 | 2780457.000710494816 | 4765777920 | 9.760313180160e+12 | 4765927417.599761009216 | 4765927417.599761009216 | 66001777459.200004577637 | 487210352.639999985695 |

## Size Breakdown

| N | oom_total_bits | oom_kb | oom_challenge_bits | oom_B_bits | oom_z_bits | oom_zb_bits | oom_zs_bits | oom_zm_bits | oom_f_bits |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 164931.248624128784 | 20.133209060563 | 160 | 52416 | 83731.650562801369 | 25477.262302238993 | 82981.153526729264 | 750.497036072104 | 3146.335759088417 |
| 16 | 175383.508098702034 | 21.409119640955 | 160 | 56416 | 84700.551726561374 | 27210.282831405242 | 83950.054690489269 | 750.497036072104 | 6896.6735407354 |
| 64 | 209190.863395421824 | 25.535994066824 | 160 | 68800 | 83369.514008972459 | 28368.002466692928 | 82619.016972900354 | 750.497036072104 | 28493.346919756445 |
| 128 | 237176.497044725431 | 28.952209111905 | 160 | 67424 | 85827.686774497546 | 28368.002466692928 | 85077.189738425441 | 750.497036072104 | 55396.807803534924 |
| 256 | 295247.863754608901 | 36.040998993483 | 160 | 67584 | 88350.839038157428 | 28947.545770150202 | 87600.342002085323 | 750.497036072104 | 110205.478946301257 |

## Repeat Accounting

| N | mu_a | mu_b | mu_s | mu_m | epsilon_a_tail | epsilon_b_tail | epsilon_s_tail | epsilon_m_tail | epsilon_2 | epsilon_2_log2 | joint_response_success | epsilon_g_upper | epsilon_cmp_modelled | success_denominator | repeat_bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 1.455702033122 | 2.266296906134 | 1.587686787863 | 1.455702033122 | 6.823030925631e-06 | 4.483706036843e-05 | 9.552243295884e-05 | 9.747187036616e-07 | 1.368467099534e-49 | -162.321915899403 | 0.999903502941 | 0.007953415163 | 0.078766056296 | 0.913771590668 | 8.34428179357 |
| 16 | 1.350280704371 | 2.266296906134 | 1.727175824702 | 1.455702033122 | 1.462078055492e-05 | 4.776121647942e-05 | 9.747187036616e-05 | 9.747187036616e-07 | 1.181984795926e-50 | -165.855193266368 | 0.999901553506 | 0.007791314964 | 0.08056203689 | 0.912127618885 | 8.435178948968 |
| 64 | 1.423863145128 | 2.266296906134 | 1.650153073711 | 1.455702033122 | 6.140727833068e-05 | 4.971065388674e-05 | 9.552243295884e-05 | 9.747187036616e-07 | 1.343435212209e-49 | -162.348549901095 | 0.999903502941 | 0.008953918582 | 0.09304766034 | 0.898644963727 | 8.625685740071 |
| 128 | 1.650153073711 | 2.266296906134 | 1.423863145128 | 1.455702033122 | 0.000123789275 | 4.971065388674e-05 | 9.649715166250e-05 | 9.747187036616e-07 | 4.289402350829e-50 | -163.995628095608 | 0.999902528224 | 0.007711269632 | 0.091274372163 | 0.901473880178 | 8.598617463525 |
| 256 | 1.727175824702 | 2.266296906134 | 1.350280704371 | 1.455702033122 | 0.000248553269 | 5.068537259040e-05 | 9.844658906982e-05 | 9.747187036616e-07 | 3.922950438392e-51 | -167.446393730386 | 0.999900578788 | 0.008518570968 | 0.089497535404 | 0.902386434262 | 8.526235986558 |

## Full TSV Columns

The TSV additionally includes fixed constants, selector raw/inf-norm vectors,
MR09 left/right sides, estimator block sizes, and diagnostic A-MSIS cost bits.

```text
N
d
w
gamma
q0
q0_mod_8
B_e
beta
embedded_key_rank
tail_factor
rhf_target
attempt_target
n
ell
p
p_bits
p_mod_8
q
log2_q
hat_n
hat_k
hat_q
hat_q_bits
hat_q_mod_8
hat_q_lower_bound
K_b
K_a
s_c
rej1_constant
q_tilde
profile
phi_a
phi_s
phi_m
phi_m_constraint_lhs
phi_m_constraint_rhs
phi_b
tau_g0
tau_g1
epsilon_g_upper
epsilon_cmp_modelled
compression_joint_pass_residues
B_a
mathcal_B
B_s
eta_m
B_response
sigma_s
sigma_m
beta_sis_1
beta_sis
beta_sis_bound_1_4wgamma_beta_sis_1
beta_sis_bound_2_beta_sis_1_plus_2B_response
beta_sis_active_bound
beta_sis_2
beta_sis_2_q_requirement
twelve_sigma_s
twelve_sigma_m
outer_msis_mr09_lhs
outer_msis_mr09_rhs
outer_auxiliary_msis2_delta_required
outer_auxiliary_msis2_mr09_lhs
outer_auxiliary_msis2_mr09_rhs
B_g_0
B_g_1
beta_sel_inf
selector_six_widths
selector_six_raw_bounds
selector_six_inf_bounds
selector_merged_widths
selector_merged_raw_bounds
selector_merged_inf_bounds
mu_a
mu_b
mu_s
mu_m
epsilon_a_tail
epsilon_b_tail
epsilon_s_tail
epsilon_m_tail
epsilon_2
epsilon_2_log2
epsilon_2_dimension
joint_response_success
success_denominator
repeat_bound
oom_kb
oom_total_bits
oom_challenge_bits
oom_B_bits
oom_z_bits
oom_zb_bits
oom_zs_bits
oom_zm_bits
oom_f_bits
outer_mlwr_delta
outer_mlwr_blocksize
outer_mlwr_arora_gb_bits
outer_hiding_mlwe_delta
outer_hiding_mlwe_blocksize
selector_hiding_mlwe_delta
selector_hiding_mlwe_blocksize
selector_binding_asis_delta
selector_binding_asis_blocksize
selector_binding_asis_cost_bits_diagnostic
all_checks_pass
```
