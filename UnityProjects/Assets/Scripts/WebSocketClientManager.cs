using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Events;
using WebSocketSharp;

public class WebSocketClientManager
{
    public static WebSocket webSocket;
    public static UnityAction<Dictionary<string, PlayerActionData>> recieveCompletedHandler;

    /// <summary>
    /// WebSocket接続
    /// </summary>
    public static void Connect()
    {
        if (webSocket == null)
        {
            webSocket = new WebSocket("ws://192.168.0.29:3000");
            webSocket.OnMessage += (sender, e) => RecieveAllPlayerAction(e.Data);
            webSocket.Connect();
        }
    }

    /// <summary>
    /// WebSocket切断
    /// </summary>
    public static void DisConnect()
    {
        webSocket.Close();
        webSocket = null;
    }

    /// <summary>
    /// WebSocket送信
    /// </summary>
    /// <param name="action"></param>
    /// <param name="pos"></param>
    /// <param name="way"></param>
    /// <param name="range"></param>
    public static void SendPlayerAction(string action, Vector3 pos, string way, float range)
    {
        var userActionData = new PlayerActionData
        {
            action  = action,
            way     = way,
            room_no = 1,
            user    = UserLoginData.userName,
            pos_x   = pos.x,
            pos_y   = pos.y,
            pos_z   = pos.z,
            range   = range
        };

        webSocket.Send(userActionData.ToJson());
    }

    /// <summary>
    /// WebSocket受信
    /// </summary>
    /// <param name="json"></param>
    public static void RecieveAllPlayerAction(string json)
    {
        var allUserActionHash = PlayerActionData.FromJson(json, 1);
        recieveCompletedHandler?.Invoke(allUserActionHash);
        // Debug.Log(json);
    }
}