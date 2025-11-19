test_file = r"C:\Users\SURFACE\OneDrive - University of Bolton\Assessments\7006\DISSERTATION\Wind Codes\data\midas\01125_rochdale\qc-version-1\midas-open_uk-hourly-weather-obs_dv-202507_greater-manchester_01125_rochdale_qcv-1_2010.csv"

with open(test_file, 'r') as fh:
    for i in range(20):
        print(fh.readline())