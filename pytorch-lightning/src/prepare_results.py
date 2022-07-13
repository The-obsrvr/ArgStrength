import pandas as pd


def prepare_single_task(single_task_data: pd.DataFrame, sampling_strategy: str = None):
    """

    :param sampling_strategy:
    :param single_task_data:
    :return:
    """
    single_task_data["use_topic"] = single_task_data["task_id"].str.contains("_topic")
    d = {True: 'topic', False: 'no-topic'}
    results = single_task_data.replace(d)

    datasets = ['gretz', 'swanson', 'ukp', 'webis', 'toledo']
    aggregated_cols = [col for col in results.columns if any(s in col for s in datasets)]

    #  aggregate over the seeds.
    results_by_task = results.groupby(["task_id", "sampling", "use_topic"], as_index=False).agg(
        {
            d: ['mean', 'std']
            for d in aggregated_cols
        }).round(3)

    # extract the values from the hierarchical dataframe.
    task_details = results_by_task.loc[:, (slice(None), '')]
    task_details.columns = task_details.columns.get_level_values(0)

    means = results_by_task.loc[:, (slice(None), 'mean')]
    means.columns = means.columns.get_level_values(0)

    stds = results_by_task.loc[:, (slice(None), 'std')]
    stds.columns = stds.columns.get_level_values(0)

    #  separate pearson and spearman
    pearson_means = means.filter(regex='pearson')
    pearson_stds = stds.filter(regex='pearson')
    pearson_stds.columns = "d_" + pearson_stds.columns
    pearson_data = pd.concat([task_details, pearson_means, pearson_stds], axis=1)

    spearman_means = means.filter(regex='spearman')
    spearman_stds = stds.filter(regex='spearman')
    spearman_stds.columns = "d_" + spearman_stds.columns
    spearman_data = pd.concat([task_details, spearman_means, spearman_stds], axis=1)

    #  combine the mean and std values for the pearson coefficients.
    if "gretz_pearson" in pearson_data.columns:
        pearson_data["IBMArgQ"] = [
            "$\rho: " + str(pearson_data["gretz_pearson"][i]) + " \pm " + str(
                pearson_data["d_gretz_pearson"][i]) + "$"
            for i in range(len(pearson_data))]
        spearman_data["IBMArgQ"] = [
            "$\sigma: " + str(spearman_data["gretz_spearman"][i]) + " \pm " + str(
                spearman_data["d_gretz_spearman"][i]) + "$"
            for i in range(len(spearman_data))]
    else:
        pearson_data["IBMArgQ"] = "-"
        spearman_data["IBMArgQ"] = "-"

    if "toledo_pearson" in pearson_data.columns:
        pearson_data["IBMRank"] = [
            "$\rho: " + str(pearson_data["toledo_pearson"][i]) + " \pm " + str(
                pearson_data["d_toledo_pearson"][i]) + "$"
            for i in range(len(pearson_data))]
        spearman_data["IBMRank"] = [
            "$\sigma: " + str(spearman_data["toledo_spearman"][i]) + " \pm " + str(
                spearman_data["d_toledo_spearman"][i]) + "$"
            for i in range(len(spearman_data))]
    else:
        pearson_data["IBMRank"] = "-"
        spearman_data["IBMRank"] = "-"

    if "ukp_pearson" in pearson_data.columns:
        pearson_data["UKPConvArgRank"] = [
            "$\rho: " + str(pearson_data["ukp_pearson"][i]) + " \pm " + str(
                pearson_data["d_ukp_pearson"][i]) + "$"
            for i in range(len(pearson_data))]
        spearman_data["UKPConvArgRank"] = [
            "$\sigma: " + str(spearman_data["ukp_spearman"][i]) + " \pm " + str(
                spearman_data["d_ukp_spearman"][i]) + "$"
            for i in range(len(spearman_data))]
    else:
        pearson_data["UKPConvArgRank"] = "-"
        spearman_data["UKPConvArgRank"] = "-"

    if "swanson_pearson" in pearson_data.columns:
        pearson_data["SwanRank"] = [
            "$\rho: " + str(pearson_data["swanson_pearson"][i]) + " \pm " + str(
                pearson_data["d_swanson_pearson"][i]) + "$"
            for i in range(len(pearson_data))]
        spearman_data["SwanRank"] = [
            "$\sigma: " + str(spearman_data["swanson_spearman"][i]) + " \pm " + str(
                spearman_data["d_swanson_spearman"][i]) + "$"
            for i in range(len(spearman_data))]
    else:
        pearson_data["SwanRank"] = "-"
        spearman_data["SwanRank"] = "-"

    if "webis_pearson" in pearson_data.columns:
        pearson_data["Webis"] = [
            "$\rho: " + str(pearson_data["webis_pearson"][i]) + " \pm " + str(
                pearson_data["d_webis_pearson"][i]) + "$"
            for i in range(len(pearson_data))]
        spearman_data["Webis"] = [
            "$\sigma: " + str(spearman_data["webis_spearman"][i]) + " \pm " + str(
                spearman_data["d_webis_spearman"][i]) + "$"
            for i in range(len(spearman_data))]
    else:
        pearson_data["Webis"] = "-"
        spearman_data["Webis"] = "-"

    final_pearson = pearson_data[
        ["task_id", "sampling", "use_topic", "IBMArgQ", "IBMRank", "UKPConvArgRank", "SwanRank", "Webis"]]
    final_spearman = spearman_data[
        ["task_id", "sampling", "use_topic", "IBMArgQ", "IBMRank", "UKPConvArgRank", "SwanRank", "Webis"]]

    metrics_table = pd.concat([final_pearson, final_spearman]).sort_values(
        by=["task_id", "sampling", "use_topic"]).reset_index(drop=True)

    # rename task_id to proper names.
    metrics_table["task_id"][metrics_table["task_id"].str.contains("STLAS_only_gretz") == True] = "IBMArgQ"
    metrics_table["task_id"][metrics_table["task_id"].str.contains("STLAS_only_toledo") == True] = "IBMRank"
    metrics_table["task_id"][metrics_table["task_id"].str.contains("STLAS_only_ukp") == True] = "UKPConvArgRank"
    metrics_table["task_id"][metrics_table["task_id"].str.contains("STLAS_only_swanson") == True] = "SwanRank"
    metrics_table["task_id"][metrics_table["task_id"].str.contains("STLAS_only_webis") == True] = "Webis"
    metrics_table["task_id"][metrics_table["task_id"].str.contains("STLAS_LOO_gretz") == True] = "All except IBMArgQ"
    metrics_table["task_id"][metrics_table["task_id"].str.contains("STLAS_LOO_toledo") == True] = "All except IBMRank"
    metrics_table["task_id"][metrics_table["task_id"].str.contains("STLAS_LOO_ukp") == True] = \
        "All except UKPConvArgRank"
    metrics_table["task_id"][metrics_table["task_id"].str.contains("STLAS_LOO_swanson") == True] = "All except SwanRank"
    metrics_table["task_id"][metrics_table["task_id"].str.contains("STLAS_LOO_webis") == True] = "All except Webis"
    metrics_table["task_id"][metrics_table["task_id"].str.contains("STLAS") == True] = "All"

    # custom order the task ids.
    metrics_table["task_id"] = pd.Categorical(metrics_table["task_id"], ["IBMArgQ", "IBMRank", "UKPConvArgRank",
                                                                         "SwanRank", "Webis", "All except IBMArgQ",
                                                                         "All except IBMRank",
                                                                         "All except UKPConvArgRank",
                                                                         "All except SwanRank", "All except Webis",
                                                                         "All"])

    # no need to filter table if no sampling strategy selected:
    if sampling_strategy is None:
        metrics_table.sort_values(by=["task_id", "sampling", "use_topic"], inplace=True)
        #  Define the latex table structure.
        col1 = metrics_table["task_id"].tolist()
        col2 = metrics_table["sampling"].tolist()
        col3 = metrics_table["use_topic"].tolist()
        cidx = ["IBMArgQ", "IBMRank", "UKPConvArgRank", "SwanRank", "Webis"]
        iidx = pd.MultiIndex.from_arrays([
            col1, col2, col3
        ])
        metrics_value_table = metrics_table.iloc[:, 3:]
        values_list = metrics_value_table.values.tolist()
        latex_table = pd.DataFrame(
            values_list, columns=cidx, index=iidx)
        pd.options.display.float_format = '{:,.3f}'.format
        ltx_code = latex_table.to_latex(
            caption="Single Task Learning",
            header=cidx,
            position="H",
            escape=False,
            multirow=True,
            column_format="|lll|lllll|",
        )
        print(ltx_code)
        return ltx_code

    # filter table to selected sampling strategy:
    else:
        metrics_table = metrics_table[metrics_table["sampling"].str.contains(sampling_strategy) == True]
        metrics_table.sort_values(by=["task_id", "use_topic"], inplace=True)
        #  Define the latex table structure.
        col1 = metrics_table["task_id"].tolist()
        col2 = metrics_table["use_topic"].tolist()
        cidx = ["IBMArgQ", "IBMRank", "UKPConvArgRank", "SwanRank", "Webis"]
        iidx = pd.MultiIndex.from_arrays([
            col1, col2])
        metrics_value_table = metrics_table.iloc[:, 3:]
        values_list = metrics_value_table.values.tolist()
        latex_table = pd.DataFrame(
            values_list, columns=cidx, index=iidx)
        pd.options.display.float_format = '{:,.3f}'.format
        ltx_code = latex_table.to_latex(
            caption="Single Task Learning: " + sampling_strategy,
            header=cidx,
            position="H",
            escape=False,
            multirow=True,
            column_format="|l|l|lllll|",
        )
        print(ltx_code)
        return ltx_code


def prepare_multi_task(multi_task_data: list, inference_data: pd.DataFrame, sampling_strategy: str = None):
    """

    :param sampling_strategy:
    :param multi_task_data:
    :param inference_data:
    :return:
    """
    complete_set = pd.DataFrame(
        columns=["task_id", "sampling", "use_topic", "aggregation", "IBMArgQ", "IBMRank", "UKPConvArgRank", "SwanRank",
                 "Webis"])

    #  prepare metrics when no inference is used.
    for result_set in multi_task_data:
        result_set["use_topic"] = result_set["task_id"].str.contains("_topic")
        d = {True: 'topic', False: 'no-topic'}
        results = result_set.replace(d)
        datasets = ['gretz', 'swanson', 'ukp', 'webis', 'toledo']
        aggregated_cols = [col for col in results.columns if any(s in col for s in datasets)]
        results_by_task = results.groupby(["task_id", "sampling", "use_topic"], as_index=False).agg(
            {
                d: ['mean', 'std']
                for d in aggregated_cols
            }).round(3)

        # extract the values into a new dataframe.
        task_details = results_by_task.loc[:, (slice(None), '')]
        task_details.columns = task_details.columns.get_level_values(0)
        means = results_by_task.loc[:, (slice(None), 'mean')]
        means.columns = means.columns.get_level_values(0)
        stds = results_by_task.loc[:, (slice(None), 'std')]
        stds.columns = stds.columns.get_level_values(0)

        #  separate pearson and spearman metrics
        pearson_means = means.filter(regex='pearson')
        pearson_stds = stds.filter(regex='pearson')
        pearson_stds.columns = "d_" + pearson_stds.columns
        pearson_data = pd.concat([task_details, pearson_means, pearson_stds], axis=1)
        spearman_means = means.filter(regex='spearman')
        spearman_stds = stds.filter(regex='spearman')
        spearman_stds.columns = "d_" + spearman_stds.columns
        spearman_data = pd.concat([task_details, spearman_means, spearman_stds], axis=1)

        #  combine the mean and std values for the pearson coefficients.
        if "gretz_pearson" in pearson_data.columns:
            pearson_data["IBMArgQ"] = [
                "$\rho: " + str(pearson_data["gretz_pearson"][i]) + " \pm " + str(
                    pearson_data["d_gretz_pearson"][i]) + "$"
                for i in range(len(pearson_data))]
            spearman_data["IBMArgQ"] = [
                "$\sigma: " + str(spearman_data["gretz_spearman"][i]) + " \pm " + str(
                    spearman_data["d_gretz_spearman"][i]) + "$"
                for i in range(len(spearman_data))]
        else:
            pearson_data["IBMArgQ"] = "-"
            spearman_data["IBMArgQ"] = "-"

        if "toledo_pearson" in pearson_data.columns:
            pearson_data["IBMRank"] = [
                "$\rho: " + str(pearson_data["toledo_pearson"][i]) + " \pm " + str(
                    pearson_data["d_toledo_pearson"][i]) + "$"
                for i in range(len(pearson_data))]
            spearman_data["IBMRank"] = [
                "$\sigma: " + str(spearman_data["toledo_spearman"][i]) + " \pm " + str(
                    spearman_data["d_toledo_spearman"][i]) + "$"
                for i in range(len(spearman_data))]
        else:
            pearson_data["IBMRank"] = "-"
            spearman_data["IBMRank"] = "-"

        if "ukp_pearson" in pearson_data.columns:
            pearson_data["UKPConvArgRank"] = [
                "$\rho: " + str(pearson_data["ukp_pearson"][i]) + " \pm " + str(pearson_data["d_ukp_pearson"][i]) + "$"
                for i in range(len(pearson_data))]
            spearman_data["UKPConvArgRank"] = [
                "$\sigma: " + str(spearman_data["ukp_spearman"][i]) + " \pm " + str(
                    spearman_data["d_ukp_spearman"][i]) + "$"
                for i in range(len(spearman_data))]
        else:
            pearson_data["UKPConvArgRank"] = "-"
            spearman_data["UKPConvArgRank"] = "-"

        if "swanson_pearson" in pearson_data.columns:
            pearson_data["SwanRank"] = [
                "$\rho: " + str(pearson_data["swanson_pearson"][i]) + " \pm " + str(
                    pearson_data["d_swanson_pearson"][i]) + "$"
                for i in range(len(pearson_data))]
            spearman_data["SwanRank"] = [
                "$\sigma: " + str(spearman_data["swanson_spearman"][i]) + " \pm " + str(
                    spearman_data["d_swanson_spearman"][i]) + "$"
                for i in range(len(spearman_data))]
        else:
            pearson_data["SwanRank"] = "-"
            spearman_data["SwanRank"] = "-"

        if "webis_pearson" in pearson_data.columns:
            pearson_data["Webis"] = [
                "$\rho: " + str(pearson_data["webis_pearson"][i]) + " \pm " + str(
                    pearson_data["d_webis_pearson"][i]) + "$"
                for i in range(len(pearson_data))]
            spearman_data["Webis"] = [
                "$\sigma: " + str(spearman_data["webis_spearman"][i]) + " \pm " + str(
                    spearman_data["d_webis_spearman"][i]) + "$"
                for i in range(len(spearman_data))]
        else:
            pearson_data["Webis"] = "-"
            spearman_data["Webis"] = "-"

        final_pearson = pearson_data[
            ["task_id", "sampling", "use_topic", "IBMArgQ", "IBMRank", "UKPConvArgRank", "SwanRank", "Webis"]]
        final_spearman = spearman_data[
            ["task_id", "sampling", "use_topic", "IBMArgQ", "IBMRank", "UKPConvArgRank", "SwanRank", "Webis"]]

        metrics_table = pd.concat([final_pearson, final_spearman]).sort_values(by=["task_id", "sampling", "use_topic"])
        metrics_table["aggregation"] = "None"
        complete_set = pd.concat([complete_set, metrics_table]).reset_index(drop=True)

    # prepare inference table
    inference_data.rename(columns={"task_name": "task_id", "aggregation_method": "aggregation"}, inplace=True)

    # define topic information
    inference_data["use_topic"] = inference_data["task_id"].str.contains("_topic")
    d = {True: 'topic', False: 'no-topic'}
    results = inference_data.replace(d)
    pearson_data = results[["task_id", "sampling", "use_topic",
                            "aggregation", "gretz_pearson", "toledo_pearson",
                            "ukp_pearson", "swanson_pearson", "webis_pearson"]]
    spearman_data = results[["task_id", "sampling", "use_topic",
                             "aggregation", "gretz_spearman", "toledo_spearman",
                             "ukp_spearman", "swanson_spearman", "webis_spearman"]]

    # separate out pearson and spearman and rewrite
    if "gretz_pearson" in pearson_data.columns:
        pearson_data["IBMArgQ"] = [
            "$\rho: " + str(pearson_data["gretz_pearson"][i]) + "$"
            for i in range(len(pearson_data))]
        spearman_data["IBMArgQ"] = [
            "$\sigma: " + str(spearman_data["gretz_spearman"][i]) + "$"
            for i in range(len(spearman_data))]
    else:
        pearson_data["IBMArgQ"] = "-"
        spearman_data["IBMArgQ"] = "-"

    if "toledo_pearson" in pearson_data.columns:
        pearson_data["IBMRank"] = [
            "$\rho: " + str(pearson_data["toledo_pearson"][i]) + "$"
            for i in range(len(pearson_data))]
        spearman_data["IBMRank"] = [
            "$\sigma: " + str(spearman_data["toledo_spearman"][i]) + "$"
            for i in range(len(spearman_data))]
    else:
        pearson_data["IBMRank"] = "-"
        spearman_data["IBMRank"] = "-"

    if "ukp_pearson" in pearson_data.columns:
        pearson_data["UKPConvArgRank"] = [
            "$\rho: " + str(pearson_data["ukp_pearson"][i]) + "$"
            for i in range(len(pearson_data))]
        spearman_data["UKPConvArgRank"] = [
            "$\sigma: " + str(spearman_data["ukp_spearman"][i]) + "$"
            for i in range(len(spearman_data))]
    else:
        pearson_data["UKPConvArgRank"] = "-"
        spearman_data["UKPConvArgRank"] = "-"

    if "swanson_pearson" in pearson_data.columns:
        pearson_data["SwanRank"] = [
            "$\rho: " + str(pearson_data["swanson_pearson"][i]) + "$"
            for i in range(len(pearson_data))]
        spearman_data["SwanRank"] = [
            "$\sigma: " + str(spearman_data["swanson_spearman"][i]) + "$"
            for i in range(len(spearman_data))]
    else:
        pearson_data["SwanRank"] = "-"
        spearman_data["SwanRank"] = "-"

    if "webis_pearson" in pearson_data.columns:
        pearson_data["Webis"] = [
            "$\rho: " + str(pearson_data["webis_pearson"][i]) + "$"
            for i in range(len(pearson_data))]
        spearman_data["Webis"] = [
            "$\sigma: " + str(spearman_data["webis_spearman"][i]) + "$"
            for i in range(len(spearman_data))]
    else:
        pearson_data["Webis"] = "-"
        spearman_data["Webis"] = "-"

    final_pearson = pearson_data[
        ["task_id", "sampling", "use_topic", "aggregation", "IBMArgQ", "IBMRank", "UKPConvArgRank", "SwanRank",
         "Webis"]]
    final_spearman = spearman_data[
        ["task_id", "sampling", "use_topic", "aggregation", "IBMArgQ", "IBMRank", "UKPConvArgRank", "SwanRank",
         "Webis"]]

    infer_table = pd.concat([final_pearson, final_spearman]).sort_values(
        by=["task_id", "sampling", "use_topic", "aggregation"]).reset_index(drop=True)

    mtl_table = pd.concat([complete_set, infer_table]).sort_values(
        by=["task_id", "sampling", "use_topic", "aggregation"]).reset_index(drop=True)
    #  rename task names
    mtl_table["task_id"][mtl_table["task_id"].str.contains("MTLAS_LOO_gretz") == True] = "All except IBMArgQ"
    mtl_table["task_id"][mtl_table["task_id"].str.contains("MTLAS_LOO_toledo") == True] = "All except IBMRank"
    mtl_table["task_id"][mtl_table["task_id"].str.contains("MTLAS_LOO_ukp") == True] = "All except UKPConvArgRank"
    mtl_table["task_id"][mtl_table["task_id"].str.contains("MTLAS_LOO_swanson") == True] = "All except SwanRank"
    mtl_table["task_id"][mtl_table["task_id"].str.contains("MTLAS_LOO_webis") == True] = "All except Webis"
    mtl_table["task_id"][mtl_table["task_id"].str.contains("MTLAS") == True] = "All"
    # custom order the task ids.
    mtl_table["task_id"] = pd.Categorical(mtl_table["task_id"], ["All", "All except IBMArgQ",
                                                                 "All except IBMRank", "All except UKPConvArgRank",
                                                                 "All except SwanRank", "All except Webis",])

    if sampling_strategy is None:
        mtl_table.sort_values(by=["task_id", "sampling", "use_topic", "aggregation"], inplace=True)
        #  Define the latex table structure.
        col1 = mtl_table["task_id"].tolist()
        col2 = mtl_table["sampling"].tolist()
        col3 = mtl_table["use_topic"].tolist()
        col4 = mtl_table["aggregation"].tolist()
        cidx = ["IBMArgQ", "IBMRank", "UKPConvArgRank", "SwanRank", "Webis"]
        iidx = pd.MultiIndex.from_arrays([
            col1, col2, col3, col4
        ])
        mtl_metrics_table = mtl_table.iloc[:, 4:]
        values_list = mtl_metrics_table.values.tolist()
        mtl_latex_table = pd.DataFrame(
            values_list, columns=cidx, index=iidx)

        ltx_code = mtl_latex_table.to_latex(
            caption="Multi Task Learning",
            longtable=False,
            header=["IBMArgQ", "IBMRank", "UKPConvArgRank", "SwanRank", "Webis"],
            position="H",
            escape=False,
            multirow=True,
            float_format="%.3f",
            column_format="|l|l|l|l|lllll|",
        )
        print(ltx_code)
        return ltx_code

    else:
        mtl_table = mtl_table[mtl_table["sampling"].str.contains(sampling_strategy) == True]
        mtl_table.sort_values(by=["task_id", "sampling", "use_topic", "aggregation"], inplace=True)
        #  Define the latex table structure.
        col1 = mtl_table["task_id"].tolist()
        col3 = mtl_table["use_topic"].tolist()
        col4 = mtl_table["aggregation"].tolist()
        cidx = ["IBMArgQ", "IBMRank", "UKPConvArgRank", "SwanRank", "Webis"]
        iidx = pd.MultiIndex.from_arrays([
            col1, col3, col4
        ])
        mtl_metrics_table = mtl_table.iloc[:, 4:]
        values_list = mtl_metrics_table.values.tolist()
        mtl_latex_table = pd.DataFrame(
            values_list, columns=cidx, index=iidx)

        ltx_code = mtl_latex_table.to_latex(
            caption="Multi Task Learning: " + sampling_strategy,
            longtable=False,
            header=["IBMArgQ", "IBMRank", "UKPConvArgRank", "SwanRank", "Webis"],
            position="H",
            escape=False,
            multirow=True,
            float_format="%.3f",
            column_format="|l|l|l|lllll|",
        )
        print(ltx_code)
        return ltx_code


if __name__ == "__main__":

    prep_single_task_table = False
    prep_multi_task_table = True
    eval_results = pd.read_csv("evaluation_results.csv", sep=",")

    # Single Task Learning
    if prep_single_task_table:
        single_task_results = eval_results[eval_results["task_id"].str.contains("MTLAS") == False]
        prepare_single_task(single_task_results, sampling_strategy="balanced")

    # Multi Task Learning
    if prep_multi_task_table:
        multi_task_results = eval_results[eval_results["task_id"].isin(["MTLAS", "MTLAS_topic"])].reset_index(drop=True)
        mtlas_loo_gretz = pd.read_csv("eval_mtlas_loo_gretz.csv", sep=";")
        mtlas_loo_toledo = pd.read_csv("eval_mtlas_loo_toledo.csv", sep=";")
        mtlas_loo_swanson = pd.read_csv("eval_mtlas_loo_swanson.csv", sep=";")
        mtlas_loo_ukp = pd.read_csv("eval_mtlas_loo_ukp.csv", sep=";")
        mtlas_loo_webis = pd.read_csv("eval_mtlas_loo_webis.csv", sep=";")
        inference_results = pd.read_csv("infer_results1.csv", sep=";")
        multi_tasks_data_list = [multi_task_results, mtlas_loo_gretz, mtlas_loo_toledo,
                                 mtlas_loo_swanson, mtlas_loo_ukp, mtlas_loo_webis]
        prepare_multi_task(multi_task_data=multi_tasks_data_list,
                           inference_data=inference_results, sampling_strategy="balanced")
        # need to adjust to register longtable instead.
