

def run_model_grd():
    _, grd, weather, exog = get_data_from_database() # soufiane
    
    latest_year, latest_month, use_mutli_run = get_latest_year_status(grd) # soufiane


    if not use_mutli_run:
        df_results_st = run_stf_srm(grd, weather, exog, latest_year = latest_year)
        df_results_lt = run_ltf_srm(grd, weather, exog, latest_year = latest_year)

    else:
        df_results_st = run_stf_srm(grd, weather, exog, latest_year = latest_year - 1)
        df_results_st = correct_with_existant(df_results_st, grd) # mohammed


        grd = append_forecast_to_grd(grd, df_results_st) # soufiane

        df_results_st_2 = run_stf_srm(grd, weather, exog, latest_year = latest_year)
        df_results_st = pd.concat([df_results_st, df_results_st_2], axis=0)

        df_results_lt = run_ltf_srm(grd, weather, exog, latest_year = latest_year)

    return df_results_st, df_results_lt