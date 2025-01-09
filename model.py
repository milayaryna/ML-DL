    # # Data evaluation
    # _, mse, mse2 = FE_model.evaluate(test_x_cnn, test_x_leaves, test_x_leaves2, test_y_target1_cls, test_y_target1, test_y_target2)
    # print('-'*50)
    # print('* Summary of model performance metrics on testing data')
    # print('Target 1 MSE: {}'.format(mse))
    # print('Target 2 MSE: {}'.format(mse2))

    # Weight data evaluation
    target1_mse, target1_mae, target2_mse, target2_mae = \
        FE_model.weighted_evaluate(test_x_cnn, test_x_leaves, test_x_leaves2, test_y_target1_cls, test_y_target1, test_y_target2, use_city_row_index)

    # Return predict value (use for data post-processing)
    y_pred_cls, y_pred_reg, y_pred_reg2, \
    c_attention_list, s_attention_list, cnn_filter, tree_attention_list = FE_model.predict(test_x_cnn, test_x_leaves, test_x_leaves2)

    return y_pred_cls, y_pred_reg, y_pred_reg2, c_attention_list, s_attention_list, cnn_filter, tree_attention_list, target1_mse, target1_mae, target2_mse, target2_mae


def FECNN_train_split_grid_data(data, use_city_row_index, load_model_time_record=None):
    # Set the path to load model
    if load_model_time_record is None:
        load_model_time_record = CURRENT_YEAR_MONTH

    x_cnn, x_tree, y_target1, y_target2, \
    test_x_cnn, test_x_tree, test_y_target1, test_y_target2 = data
    training_dataset = (x_cnn, x_tree, y_target1, y_target2)
    testing_dataset = (test_x_cnn, test_x_tree, test_y_target1, test_y_target2)

    # FECNN model training
    FECNN_train_full_data(training_dataset, load_model_time_record)

    # Testing dataset evaluation
    _, _, _, _, _, _, _, \
    _, _, _, _ = FECNN_eval(testing_dataset, load_model_time_record, use_city_row_index)

def FECNN_pred(data, load_model_time_record):
    test_x_cnn, test_x_tree = data

    # Load model
    model_folder_path = join(CWD, 'save', 'model', load_model_time_record)
    FE_model = torch.load(join(model_folder_path, 'FECNN_model.pt'))

    # Transform test sample to leaves node
    test_x_leaves, test_x_leaves2 = get_bank_xgb_leaves(test_x_tree, join(model_folder_path, 'tree'))

    # Model prediction
    y_pred_cls, y_pred_reg, y_pred_reg2, \
    c_attention_list, s_attention_list, cnn_filter, tree_attention_list = FE_model.predict(test_x_cnn, test_x_leaves, test_x_leaves2)

    return y_pred_cls, y_pred_reg, y_pred_reg2, c_attention_list, s_attention_list, cnn_filter, tree_attention_list
