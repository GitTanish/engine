################################################################################
# HEAVY TACTICAL DRONE CONSTANTS
################################################################################

GRAVITY = 9.8

# Mass in kg
HEAVY_DRONE_MASS = 8.0

# Hover RPM
HEAVY_DRONE_HOVER_RPM = 5000

# Force Constant (KF)
# Calculated as: (MASS * G) / (4 * HOVER_RPM^2)
# (8.0 * 9.8) / (4 * 5000^2) = 78.4 / 100,000,000 = 7.84e-7
HEAVY_DRONE_KF = 7.84e-7

# Moment Constant (KM)
# Assumed to be 0.05 * KF
HEAVY_DRONE_KM = 3.92e-8
