import torch
nbrtuModel = torch.hub.load('yolov5', 'custom', path='AI_Models/best.pt', source='local') # , device=0)
# old_nbrtuModel = torch.hub.load('yolov5', 'custom', path='AI_Models/Nagad_AI_V3.pt', source='local', device=0)
# nbrtuModel.conf = 0.4
# nbrtuModel.iou = 0.2

NBRTU_val = [
                "nagad_festoon_nagad_sheba",
                "nagad_festoon_bmet",
                "nagad_festoon_indian_visa",
                "nagad_shop_banner_customized",
                "nagad_shop_banner_nagad_sheba",
                "nagad_sticker_running",
                "nagad_sticker_shutter",
                "nagad_sticker_table_3",
                "nagad_sticker_table_2",
                "nagad_sticker_table_1",
                "nagad_sticker_customized_table",
                "nagad_sticker_remittance_1_large",
                "nagad_sticker_remittance_2",
                "nagad_sticker_fraud_awareness",
                "nagad_sticker_qr_code",
                "nagad_sticker_hotline",
                "nagad_sticker_hotline_file_folder",
                "nagad_identifier_nagad",
                "nagad_board_back_lit_1_with_picture",
                "nagad_back_lit_2",
                "nagad_table_top_qr_wooden_base_old",
                "nagad_table_top_qr_wooden_base_new",
                "nagad_table_top_qr_pvc",
                "nagad_sticker_qr_horizontal_new",
                "nagad_sticker_qr_horizontal_old",
                "nagad_sticker_qr_vertical",
                "nagad_sticker_push_pull_door",
                "nagad_board_open_close_pvc",
                "nagad_sticker_payment",
                "nagad_sticker_payment_accepted_here",
                "nagad_hanging_display",
                "nagad_x_banner",
                "nagad_ms_standee",
                'nagad_sticker_shutter_panel', # new v5
                "rocket_sticker_qr_code", 
                "rocket_sticker_shutter",
                "rocket_sticker_running",
                "rocket_sticker_table",
                "rocket_sticker_table_price",
                "rocket_table_top_qr_1_pvc",
                "tap_table_top_qr",
                "bkash_sticker_table",
                "bkash_sticker_bill_pay_table",
                "bkash_sticker_running",
                "bkash_sticker_qr_code",
                "bkash_sticker_transaction",
                "bkash_sticker_price",
                "bkash_sticker_mask_bangla",
                "bkash_sticker_fraud_awareness",
                "bkash_sticker_shutter",
                "bkash_board_back_lit_1_with_image",
                "bkash_table_top_qr_pvc_running",
                "bkash_table_top_qr_pvc_dai_cut",
                "bkash_table_top_qr_smart",
                "bkash_sticker_qr_vertical_1",
                "bkash_sticker_payment_running",
                "bkash_sticker_payment_dai_cut",
                "bkash_sticker_payment_large",
                "bkash_sticker_push-pull_door",
                "bkash_sticker_mask_english",
                "bkash_hanger_push-pull_door",
                "bkash_pos_pointer_back_led",
                "bkash_hanging_board_open-close",
                "upay_sticker_shutter",
                "upay_sticker_qr_code_1",
                "upay_sticker_qr_code_2"                
            ]



nagad_items = [
                'nagad_festoon_indian_visa',
                'nagad_festoon_islamic',
                'nagad_festoon_lenden_1',
                'nagad_festoon_lenden_2',
                'nagad_festoon_price_1',
                'nagad_festoon_price_2',
                'nagad_festoon_price_3', # conflicted item
                'nagad_festoon_remittance',
                'nagad_hanging_display',
                'nagad_ms_standee',
                'nagad_poster_fraud_awareness_1',
                'nagad_poster_fraud_awareness_2',
                'nagad_poster_islamic',
                'nagad_poster_mobile_recharge_shakib',
                'nagad_poster_remittance_2',
                'nagad_shop_banner_lenden',
                'nagad_shop_banner_price_1',
                'nagad_shop_banner_price_2',
                'nagad_sticker_customized_table',
                'nagad_sticker_fraud_awareness',
                'nagad_sticker_hotline',
                'nagad_sticker_hotline_file_folder',
                'nagad_sticker_payment',
                'nagad_sticker_payment_accepted_here',
                'nagad_sticker_push_pull_door',
                'nagad_sticker_qr_code',
                'nagad_sticker_qr_horizontal_new',
                'nagad_sticker_qr_horizontal_old',
                'nagad_sticker_qr_vertical',
                'nagad_sticker_remittance_1_large',
                'nagad_sticker_remittance_2',
                'nagad_sticker_running',
                'nagad_sticker_shutter',
                'nagad_sticker_table_1',
                'nagad_sticker_table_2',
                'nagad_sticker_table_3',
                'nagad_table_top_qr_pvc',
                'nagad_table_top_qr_wooden_base_new',
                'nagad_table_top_qr_wooden_base_old',
                'nagad_shop_banner_cash_out', # new v5
                'nagad_festoon_send_money_free', # new v5
                'nagad_poster_pdb', # new v5
                'nagad_sticker_shutter_panel', # new v5
                'nagad_poster_all_service' # new v5
            ]

ndel_items = [
                'nagad_festoon_bmet',
                'nagad_board_open_close_pvc',
                'nagad_identifier_nagad',
                'nagad_poster_1000_cashback',
                'nagad_poster_ananto_bmw',
                'nagad_table_talker',
                'nagad_sticker_payment_bangla',
                'nagad_shop_banner_nagad_sheba',
                'nagad_festoon_nagad_sheba',
                'nagad_poster_mobile_recharge',
                'nagad_back_lit_2',
                'nagad_board_back_lit_1_with_picture',
                'nagad_x_banner',
                'bkash_table_top_qr_smart',
                'nagad_festoon_send_money', # new add v5
                'nagad_shop_banner_customized', # new add v5
                'tap_table_top_qr', # add new v5
                'bkash_hanger_push-pull_door' # add new v5
            ]



bkash_items = [   
                'bkash_festoon_bill_pay', 
                'bkash_festoon_digital', 
                'bkash_festoon_price', # updated v4 added yellow also
                'bkash_festoon_qr', 
                'bkash_festoon_remittance', 
                'bkash_festoon_savings', 
                'bkash_hanging_board_open-close',
                'bkash_pos_pointer_back_led', 
                'bkash_poster_islamic', 
                'bkash_poster_mobile_recharge', 
                'bkash_poster_remittance', 
                'bkash_poster_savings', 
                'bkash_festoon_savings_new' # new v4
                'bkash_poster_send_money', 
                'bkash_poster_table_messi', 
                'bkash_poster_western_union', 
                'bkash_shop_banner_price', # updated v4 added yellow also
                'bkash_sticker_bill_pay_table', 
                'bkash_sticker_fraud_awareness', 
                'bkash_sticker_mask_bangla', 
                'bkash_sticker_mask_english', 
                'bkash_sticker_payment_dai_cut', 
                'bkash_sticker_payment_large', 
                'bkash_sticker_payment_running', 
                'bkash_sticker_price', 
                'bkash_sticker_push-pull_door',
                'bkash_sticker_qr_code', 
                'bkash_sticker_qr_vertical_1', 
                'bkash_sticker_running', 
                'bkash_sticker_shutter', 
                'bkash_sticker_table', 
                'bkash_sticker_transaction', 
                'bkash_table_top_qr_pvc_dai_cut', 
                'bkash_table_top_qr_pvc_running'
            ]




rocket_items = [
                    'rocket_sticker_qr_code', 
                    'rocket_festoon_bill_pay', 
                    'rocket_festoon_lenden', 
                    'rocket_festoon_price', 
                    'rocket_festoon_send_money', 
                    'rocket_poster_add_money', 
                    'rocket_poster_insurance', 
                    'rocket_shop_banner_bill_pay', 
                    'rocket_shop_banner_lenden', 
                    'rocket_shop_banner_price', 
                    'rocket_shop_banner_send_money', 
                    'rocket_sticker_running', 
                    'rocket_sticker_shutter', 
                    'rocket_sticker_table', 
                    'rocket_sticker_table_price', 
                    'rocket_table_top_qr_1_pvc',
                    'rocket_poster_isnsurance',
                    'rocket_fastoon_wasa_bill', # new v5
                    'rocket_fastoon_bill_pay_new', # new v5
                    'rocket_fastoon_pdb_bill', # new v5
                    'rocket_fastoon_lenden_new', # new v5
                    'rocket_fastoon_polli_biddut', # new v5
                    'rocket_fastoon_remittance' # new v5
                ]



tap_items = [
                'tap_table_top_qr'
            ]



upay_items = [
                'upay_festoon_price', 
                'upay_festoon_traffic_1', 
                'upay_festoon_traffic_2', 
                'upay_poster_lenden', 
                'upay_shop_banner', 
                'upay_sticker_qr_code_1', 
                'upay_sticker_qr_code_2', 
                'upay_sticker_shutter',
                'upay_table_top_qr_pvc',
                'upay_fastoon_dpdc', # new v5
                'upay_fastoon_price_new', # new v5
                'upay_fastoon_water_bill' # new v5
            ]



# new_items = [
#                 'nagad_festoon_price_3', # conflicted item
#                 'nagad_shop_banner_cash_out', # new v5
#                 'nagad_festoon_send_money_free', # new v5
#                 'nagad_poster_pdb', # new v5
#                 'nagad_sticker_shutter_panel', # new v5
#                 'nagad_poster_all_service', # new v5
#                 'bkash_festoon_savings_new' # new v4
#                 'bkash_festoon_price', # updated v4 added yellow also
#                 'bkash_shop_banner_price', # updated v4 added yellow also
#                 'rocket_fastoon_wasa_bill', # new v5
#                 'rocket_fastoon_bill_pay_new', # new v5
#                 'rocket_fastoon_pdb_bill', # new v5
#                 'rocket_fastoon_lenden_new', # new v5
#                 'rocket_fastoon_polli_biddut', # new v5
#                 'rocket_fastoon_remittance' # new v5
#                 'upay_fastoon_dpdc', # new v5
#                 'upay_fastoon_price_new', # new v5
#                 'upay_fastoon_water_bill' # new v5
#             ]