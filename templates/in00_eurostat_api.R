
# Access the Eurostat Database from R
# See https://ropengov.github.io/eurostat/articles/eurostat_tutorial.html

library(eurostat)
library(dplyr)

# Get Eurostat data listing
toc <- get_eurostat_toc()

#-------------------------------------------------------------------------------
# Overview
toc %>% group_by(type) %>% count()
# # A tibble: 3 × 2
# # Groups:   type [3]
# type        n
# <chr>   <int>
# 1 dataset  7590
# 2 folder   1886
# 3 table    1461

#-------------------------------------------------------------------------------
# Search the db
search_eurostat("transaction") %>% 
  bind_rows(search_eurostat("financial")) %>%
  bind_rows(search_eurostat("bank")) %>%
  distinct() %>% select(title, contains("last update")) %>%
  print(n = nrow(.))

search_eurostat("credit card")
search_eurostat("payments")
search_eurostat("account")

search_eurostat("inancial transactions") %>% 
  select(title, code, contains("last update"))

# A tibble: 9 × 3
# title                                                                    code            `last update of data`
# <chr>                                                                    <chr>           <chr>                
# 1 Non-financial transactions - quarterly data                              nasq_10_nf_tr   11.01.2023           
# 2 Non-financial transactions - annual data                                 nasa_10_nf_tr   11.01.2023           
# 3 Financial transactions - annual data                                     nasa_10_f_tr    12.01.2023           
# 4 Non-financial transactions - quarterly data                              nasq_10_nf_tr   11.01.2023           
# 5 Financial transactions - quarterly data                                  nasq_10_f_tr    14.01.2023           
# 6 Non-financial transactions - selected international annual data          naidsa_10_nf_tr 11.01.2023           
# 7 Financial transactions – international data cooperation                  naidsa_10_f_tr  20.12.2022           
# 8 Non-financial transactions - selected international quarterly data       naidsq_10_nf_tr 11.01.2023           
# 9 Financial transactions – international data cooperation - quarterly data naidsq_10_f_tr  14.01.2023     

# -> most interesting:  nasq_10_f_tr
dir.create("../../../help")
dat <- get_eurostat("nasq_10_f_tr")

# Save the data locally
catch = function(cmd) tryCatch(cmd, 
                 warning = function(cond) print("warning"),
                 error = function(cond) print("error"),
                 finally = NULL
                 )
dir.create("../../../data") %>% catch()
dir.create("../../../data/eurostat") %>% catch()

qs::qsave(dat, file = "../../../data/eurostat/nasq_10_f_tr_23-01-14.qs")



dat <- qs::qread("../../../data/eurostat/nasq_10_f_tr_23-01-14.qs")

# Labels
label_eurostat_vars(names(dat))
# [1] "Unit of measure"                                                 
# [2] "Sector"                                
# [3] "Financial position"                    
# [4] "National accounts indicator (ESA 2010)"
# [5] "Geopolitical entity (reporting)"       
# [6] "Period of time"    

dat %>% group_by(geo) %>% count()
dat %>% group_by(finpos) %>% count()
dat %>% group_by(sector) %>% count()
dat %>% group_by(unit) %>% count()


