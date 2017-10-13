"""
const.py

Just a bunch of constants.
"""

MODEL_CONFIG_CONTENT = '<?xml version="1.0" ?>\n\n' \
                       '<model>\n' \
                       '\t<name>{}</name>\n' \
                       '\t<version>1.0</version>\n' \
                       '\t<sdf version="1.5">model.sdf</sdf>\n\n' \
                       '\t<description></description>\n' \
                       '</model>\n'  # format with the name of the model

MODEL_SDF_CONTENT = '<?xml version="1.0" ?>\n' \
                    '<sdf version="1.5">\n' \
                    '\t<model name="{}">\n' \
                    '\t\t<static>true</static>\n' \
                    '\t\t<link name="link">\n' \
                    '\t\t\t<gravity>0</gravity>\n' \
                    '\t\t\t<inertial>\n' \
                    '\t\t\t\t<pose>{} {} {} {} {} {}</pose>\n' \
                    '\t\t\t\t<inertia>\n' \
                    '\t\t\t\t\t<ixx>{}</ixx>\n' \
                    '\t\t\t\t\t<iyy>{}</iyy>\n' \
                    '\t\t\t\t\t<izz>{}</izz>\n' \
                    '\t\t\t\t</inertia>\n' \
                    '\t\t\t\t<mass>{}</mass>\n' \
                    '\t\t\t</inertial>\n' \
                    '\t\t\t<collision name="collision">\n' \
                    '\t\t\t\t<pose>{} {} {} {} {} {}</pose>\n' \
                    '\t\t\t\t<geometry>\n' \
                    '\t\t\t\t\t<mesh>\n' \
                    '\t\t\t\t\t\t<uri>{}</uri>\n' \
                    '\t\t\t\t\t</mesh>\n' \
                    '\t\t\t\t</geometry>\n' \
                    '\t\t\t\t<surface>\n' \
                    '\t\t\t\t\t<contact>\n' \
                    '\t\t\t\t\t\t<ode>\n' \
                    '\t\t\t\t\t\t\t<max_vel>{}</max_vel>\n' \
                    '\t\t\t\t\t\t\t<min_depth>{}</min_depth>\n' \
                    '\t\t\t\t\t\t</ode>\n' \
                    '\t\t\t\t\t</contact>\n' \
                    '\t\t\t\t</surface>\n' \
                    '\t\t\t</collision>\n' \
                    '\t\t\t<visual name="visual">\n' \
                    '\t\t\t\t<pose>{} {} {} {} {} {}</pose>\n' \
                    '\t\t\t\t<geometry>\n' \
                    '\t\t\t\t\t<mesh>\n' \
                    '\t\t\t\t\t\t<uri>{}</uri>\n' \
                    '\t\t\t\t\t\t<scale>{} {} {}</scale>\n' \
                    '\t\t\t\t\t</mesh>\n' \
                    '\t\t\t\t</geometry>\n' \
                    '\t\t\t</visual>\n' \
                    '\t\t</link>\n' \
                    '\t</model>\n' \
                    '</sdf>\n'

# (model_name, ip0, ip1, ip2, ip3, ip4, ip5, ixx, iyy, izz, mass,
# cp0, cp1, cp2, cp3, cp4, cp5, uri, max_vel, min_depth, vp0, vp1, vp2, vp3,
# vp4, vp5, uri, scale0, scale1, scale2)
