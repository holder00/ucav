local log_file = nil

package.path  = package.path..";.\\LuaSocket\\?.lua"
package.cpath = package.cpath..";.\\LuaSocket\\?.dll"
socket = require("socket")
-- host = host or "localhost"
-- port = port or 27015
host = "127.0.0.1"
port = 27015

port_recv = 27016
nojoy_cam_pose = nil
rel_pos = {}
rel_pos.x = -10
rel_pos.y = 10
rel_pos.z = 0

count = 0

function LuaExportStart()
    log_file = io.open("C:/Users/xuhao/Saved Games/DCS/Logs/EasyControl.log", "w")
    log.write("EasyControl.EXPORT", log.INFO, "initializing DCS Easy Control")
end

function LuaExportBeforeNextFrame()
end

function LuaExportAfterNextFrame()
end

function LuaExportStop()
    
    if log_file then
        log_file:close()
        log_file = nil
    end	

    if connectSoc then
        -- socket.try(connectSoc:send("quit")) -- to close the listener socket
        connectSoc:close()
    end
end

function parse_incoming_message(data, myData)
    numbers = {}
    local data_num = 0
    for num in string.gmatch(data, '([^,]+)') do
        table.insert(numbers, tonumber(num))
        data_num = data_num + 1
    end
    local time = LoGetModelTime()
    local dt = time - numbers[1]
    -- log.write("EasyControl.EXPORT", log.INFO, string.format(

    log.write("EasyControl.EXPORT", log.INFO, string.format(
        "timenow %f dt %f ctrl %f %f %f %f", 
        time, dt*1000, numbers[3], numbers[2], numbers[4], numbers[5]))

    LoSetCommand(2001, numbers[3]) -- elevator
    LoSetCommand(2002, numbers[2]) -- aileron
    LoSetCommand(2003, numbers[4]) -- rudder
    LoSetCommand(2004, -numbers[5]) -- thrust

    local cam_pose = LoGetCameraPosition()
    -- cam_pose.p.x = myData.Position.x + numbers[6]
    -- cam_pose.p.y = myData.Position.y + numbers[7]
    -- cam_pose.p.z = myData.Position.z + numbers[8]

    rel_pos.x = numbers[6]
    rel_pos.y = numbers[7]
    rel_pos.z = numbers[8]

    if data_num >= 17 then
        cam_pose.x.x = numbers[9]
        cam_pose.y.x = numbers[10]
        cam_pose.z.x = numbers[11]

        cam_pose.x.y = numbers[12]
        cam_pose.y.y = numbers[13]
        cam_pose.z.y = numbers[14]

        cam_pose.x.z = numbers[15]
        cam_pose.y.z = numbers[16]
        cam_pose.z.z = numbers[17]
        nojoy_cam_pose = cam_pose
        -- LoSetCameraPosition(cam_pose)
        return cam_pose
    end
    return nil
end

function LuaExportActivityNextEvent(t)
    if connectSoc==nil then
        -- log_file:write("try to open socket\n")
        log.write("EasyControl.EXPORT", log.INFO, "try to open socket")
        
        connectSoc = assert(socket:udp())
        assert(connectSoc:setoption('broadcast', true))
        assert(connectSoc:setoption('dontroute', true))   -- do we need this?

        if connectSoc then
            log.write("EasyControl.EXPORT", log.INFO, "socket ok")
        else
            return t+0.5
        end
    end

    if connectSocRecv==nil then
        -- log_file:write("try to open socket\n")
        log.write("EasyControl.EXPORT", log.INFO, "try to open socket")
        
        connectSocRecv = assert(socket:udp())

        connectSocRecv:setsockname("*", port_recv)
        connectSocRecv:settimeout(0)

        if connectSocRecv then
            log.write("EasyControl.EXPORT", log.INFO, "recv socket ok")
        else
            return t+0.5
        end
    end

    local myData = LoGetSelfData()
    if (myData) then
        local altBar = LoGetAltitudeAboveSeaLevel()
        local altRad = LoGetAltitudeAboveGroundLevel()
        local pitch, roll, yaw = myData.Pitch, myData.Bank, myData.Heading
        -- local pitch, roll, yaw = LoGetADIPitchBankYaw()
        local pos = myData.Position
        local _type = myData.Type
        local aoa = LoGetAngleOfAttack()
        local tas = LoGetTrueAirSpeed()
        local omega = LoGetAngularVelocity()
        local cam_pose = {}
        local time = LoGetModelTime()
        local nz = LoGetAccelerationUnits().y;
        data = connectSocRecv:receive()
        if data then
            nojoy_cam_pose = parse_incoming_message(data, myData)
        end
        if nojoy_cam_pose then
            cam_pose = nojoy_cam_pose
            cam_pose.p.x = pos.x + rel_pos.x -- + nojoy_cam_pose.p.x
            cam_pose.p.y = pos.y + rel_pos.y -- + nojoy_cam_pose.p.y
            cam_pose.p.z = pos.z + rel_pos.z -- + nojoy_cam_pose.p.z
            LoSetCameraPosition(cam_pose)
        else
            cam_pose = LoGetCameraPosition()
            cam_pose.p.x = pos.x + rel_pos.x -- + nojoy_cam_pose.p.x
            cam_pose.p.y = pos.y + rel_pos.y -- + nojoy_cam_pose.p.y
            cam_pose.p.z = pos.z + rel_pos.z -- + nojoy_cam_pose.p.z
            LoSetCameraPosition(cam_pose)
        end

        if count%5 == 0 then
            local _datalog = string.format(
                "name=%s time=%.3f altBar=%.3f x=%.5f y=%.5f z=%.5f pitch=%.5f roll=%.5f yaw=%.5f yawrate=%.5f pitchrate=%.5f rollrate=%.5f tas=%.3f aoa=%.5f Nz=%.5f\
                Rcamxx=%.5f  Rcamxy=%.5f Rcamxz=%.5f Rcamyx=%.5f  Rcamyy=%.5f  Rcamyz=%.5f  Rcamzx=%.5f  Rcamzy=%.5f Rcamzz=%.5f Tcamx=%.5f Tcamy=%.5f Tcamz=%.5f\n", 
                myData.Name, time, altBar, pos.x, pos.y, pos.z, 
                pitch, roll, yaw, omega.y, omega.z, omega.x, tas, aoa, nz,
                cam_pose.x.x, cam_pose.x.y, cam_pose.x.z, 
                cam_pose.y.x, cam_pose.y.y, cam_pose.y.z, 
                cam_pose.z.x, cam_pose.z.y, cam_pose.z.z, 
                cam_pose.p.x, cam_pose.p.y, cam_pose.p.z
            )
        
            socket.try(connectSoc:sendto(_datalog, host, port))
            log_file:write(_datalog)
        end
    end
    count = count + 1
    return t+0.001
end

-- See https://wiki.hoggitworld.com/view/DCS_Export_Script
-- Index 00 = LoGetAccelerationUnits().x = Lateral acceleration (G)
-- Index 01 = LoGetAccelerationUnits().z = Longitudinal acceleration (G)
-- Index 02 = LoGetAccelerationUnits().y = Vertical acceleration (G)
-- Index 03 = LoGetVectorVelocity().x = Lateral speed (m/s)
-- Index 04 = LoGetVectorVelocity().z = Longitudinal speed (m/s)
-- Index 05 = LoGetVectorVelocity().y = Vertical speed (m/s)
-- Index 06 = LoGetAngularVelocity().z = Rotation speed around z (pitchrate in rad/s) body frame
-- Index 07 = LoGetAngularVelocity().y = Rotation speed around y (yawrate in rad/s) body frame
-- Index 08 = LoGetAngularVelocity().x = Rotation speed around x (rollrate in rad/s) body frame 
-- Index 09 = LoGetADIPitchBankYaw(0) = Yaw position (rad)
-- Index 10 = LoGetADIPitchBankYaw(1) = Roll position (rad)
-- Index 11 = LoGetADIPitchBankYaw(2) = Pitch position (rad)
-- Index 12 = LoGetTrueAirSpeed() = Air speed (m/s)
-- Index 13 = LoGetAircraftDrawArgumentValue(1) = Front/Rear landing gear (0 to 1)?
-- Index 14 = LoGetAircraftDrawArgumentValue(2) = Turning landing gear (0 to 1)?
-- Index 15 = LoGetAircraftDrawArgumentValue(4) = Left landing gear (0 to 1)?
-- Index 16 = LoGetAircraftDrawArgumentValue(6) = Right landing gear (0 to 1)?
-- Index 17 = LoGetAltitudeAboveGroundLevel() = Vertical position relative to ground (m)
-- Index 17 = LoGetModelTime() = To track frame sequence (not used right now, in seconds)

-- local cam_pos = LoGetCameraPosition() x y z is a rotation matrix
