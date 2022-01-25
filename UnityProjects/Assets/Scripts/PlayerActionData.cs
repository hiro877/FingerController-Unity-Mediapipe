using Newtonsoft.Json;
using System.Collections.Generic;
using UnityEngine;

public class PlayerActionData
{
    [JsonProperty("action")]
    public string action;

    [JsonProperty("room_no")]
    public int? room_no;

    [JsonProperty("user")]
    public string user;

    [JsonProperty("pos_x")]
    public float pos_x;

    [JsonProperty("pos_y")]
    public float pos_y;

    [JsonProperty("pos_z")]
    public float pos_z;

    [JsonProperty("way")]
    public string way;

    [JsonProperty("range")]
    public float range;

    /// <summary>
    /// クライアントからサーバへ送信するデータをJSON形式に変換
    /// </summary>
    /// <returns></returns>
    public string ToJson()
    {
        // オブジェクトをjsonに変換
        return JsonConvert.SerializeObject(this, Formatting.None);
    }

    /// <summary>
    /// サーバーから送信してきたJSONデータを配列データに変換
    /// </summary>
    /// <param name="json"></param>
    /// <param name="roomNo"></param>
    /// <returns></returns>
    public static Dictionary<string, PlayerActionData> FromJson(string json, int roomNo)
    {
        // json文字列を多階層のDictionaryに変換
        var jsonHash = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, Dictionary<string, object>>>>(json);

        // 戻り値のDictionaryの初期化
        var playerActionHash = new Dictionary<string, PlayerActionData>();

        // jsonの中に該当のルーム番号の情報がなければ空のDictionaryを返却
        if (!jsonHash.ContainsKey("room" + roomNo))
        {
            return playerActionHash;
        }

        // ルームの中にユーザ情報が含まれているのでPlayerActionData型に変換
        var roomPlayerHash = jsonHash["room" + roomNo];
        foreach (var playerHash in roomPlayerHash)
        {
            // Debug.Log(playerHash.Value["pos_x"]);
            var PlayerActionData = new PlayerActionData
            {
                user  = (string)playerHash.Value["user"],
                pos_x = float.Parse(playerHash.Value["pos_x"].ToString()),
                pos_y = float.Parse(playerHash.Value["pos_y"].ToString()),
                pos_z = float.Parse(playerHash.Value["pos_z"].ToString()),
                way   = (string)playerHash.Value["way"],
                range = float.Parse(playerHash.Value["range"].ToString()),
            };
            playerActionHash.Add(PlayerActionData.user, PlayerActionData);
        }

        return playerActionHash;
    }
}