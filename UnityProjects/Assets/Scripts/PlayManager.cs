using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class PlayManager : MonoBehaviour
{
    private GameObject playerPrefab = null;     // プレイヤーのリソース(プレハブ)
    private GameObject player;                  // 自プレイヤー情報
    private const float KEY_MOVEMENT = 0.5f;    // 移動ボタン1回クリックでの移動量

    public float speed = 3.0f;

    // 全プレイヤーの行動情報
    private Dictionary<string, PlayerActionData> PlayerActionMap;      

    // 全プレイヤーのオブジェクト情報
    private readonly Dictionary<string, GameObject> playerObjectMap = new Dictionary<string, GameObject>();

    // Start is called before the first frame update
    void Start()
    {
        // 自プレイヤーの作成
        player = MakePlayer(Vector3.zero, UserLoginData.userName);

        // WebSocket開始
        StartWebSocket();
    }

    /// <summary>
    /// 定期更新
    /// </summary>
    void Update()
    {
        // ユーザーの行動情報があったら同期処理を行い、ユーザーの行動情報を初期化
        if (PlayerActionMap != null)
        {
            Synchronaize();
            PlayerActionMap = null;
        }
    }

    /// <summary>
    /// 上ボタン押下時の処理
    /// </summary>
    public void OnClickUpButton()
    {
        // 行動情報を送信＆移動
        WebSocketClientManager.SendPlayerAction("move", player.transform.position, "up", KEY_MOVEMENT);
        player.transform.Translate(0, 0, KEY_MOVEMENT);
    }

    /// <summary>
    /// 下ボタン押下時の処理
    /// </summary>
    public void OnClickDownButton()
    {
        // 行動情報を送信＆移動
        WebSocketClientManager.SendPlayerAction("move", player.transform.position, "down", KEY_MOVEMENT);
        player.transform.Translate(0, 0, -KEY_MOVEMENT);    }

    /// <summary>
    /// 左ボタン押下時の処理
    /// </summary>
    public void OnClickLeftButton()
    {
        // 行動情報を送信＆移動
        WebSocketClientManager.SendPlayerAction("move", player.transform.position, "left", KEY_MOVEMENT);
        player.transform.Translate(-KEY_MOVEMENT, 0, 0);    }

    /// <summary>
    /// 右ボタン押下時の処理
    /// </summary>
    public void OnClickRightButton()
    {
        // 行動情報を送信＆移動
        WebSocketClientManager.SendPlayerAction("move", player.transform.position, "right", KEY_MOVEMENT);
        player.transform.Translate(KEY_MOVEMENT, 0, 0);    }

    /// <summary>
    /// 退室ボタン押下時の処理
    /// </summary>
    public void OnClickExitButton()
    {
        // WebSocket通信終了
        EndWebsocket();

        // タイトルシーンに戻る
        SceneManager.LoadScene("TitleScene");
    }

    /// <summary>
    /// WebSocketの開始
    /// </summary>
    private void StartWebSocket()
    {
        // WebSocket通信開始
        WebSocketClientManager.Connect();

        // WebSocketのメッセージ受信メソッドの設定
        WebSocketClientManager.recieveCompletedHandler += OnReciveMessage;

        // 自プレイヤーの初期情報をWebSocketに送信
        WebSocketClientManager.SendPlayerAction("connect", Vector3.zero, "neutral", 0.0f);
    }

    /// <summary>
    /// WebSocketの終了
    /// </summary>
    private void EndWebsocket()
    {
        WebSocketClientManager.SendPlayerAction("disconnect", Vector3.zero, "neutral", 0.0f);
        WebSocketClientManager.DisConnect();
    }

    /// <summary>
    ///  WebSocketのメッセージ(ユーザーの行動情報)受信メソッド
    /// </summary>
    /// <param name="synchronizeData"></param>
    private void OnReciveMessage(Dictionary<string, PlayerActionData> PlayerActionMap)
    {
        // 同期情報を取得
        this.PlayerActionMap = PlayerActionMap;
    }

    /// <summary>
    /// 同期処理
    /// </summary>
    private void Synchronaize()
    {

        // 退出した他プレイヤーの検索
        List<string> otherPlayerNameList = new List<string>(playerObjectMap.Keys);
        foreach (var otherPlayerName in otherPlayerNameList)
        {
            // 退出したプレイヤーの削除
            if (!PlayerActionMap.ContainsKey(otherPlayerName))
            {
                Destroy(playerObjectMap[otherPlayerName]);
                playerObjectMap.Remove(otherPlayerName);
            }
        }

        // プレイヤーの位置を更新
        foreach (var playerAction in PlayerActionMap.Values)
        {
            // 自分は移動済みなのでスルー
            if (UserLoginData.userName == playerAction.user)
            {
                continue;
            }

            // 入室中の他プレイヤーの移動
            if (playerObjectMap.ContainsKey(playerAction.user))
            {
                if(playerAction.way == "mediapipe"){
                    playerObjectMap[playerAction.user].transform.position = playerObjectMap[playerAction.user].transform.position + GetFingerMovePos(playerAction);
                } else {
                    playerObjectMap[playerAction.user].transform.position = GetMovePos(playerAction);
                }

            // 入室中した他プレイヤーの生成
            } 
            else
            {
                // 他プレイヤーの作成
                var player = MakePlayer(GetMovePos(playerAction), playerAction.user);

                // 他プレイヤーリストへの追加
                playerObjectMap.Add(playerAction.user, player);
            }
        }
    }

    /// <summary>
    /// プレイヤーを作成
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="name"></param>
    private GameObject MakePlayer(Vector3 pos, string name)
    {

        string playerPrefabName = "";
        if(name.Equals("mediapipe")){
            playerPrefabName = "MediapipePlayer";
        } else {
            playerPrefabName = "SphPlayer";
        }

        playerPrefab = (GameObject)Resources.Load(playerPrefabName);        

        // プレイヤーを生成
        var player = (GameObject)Instantiate(playerPrefab, pos, Quaternion.identity);

        // プレイヤーのネームプレートの設定
        var otherNameText = player.transform.Find(playerPrefabName).gameObject;
        otherNameText.GetComponent<TextMesh>().text = name;

        return player;
    }

    /// <summary>
    /// 各プレイヤーの移動後の座標を取得
    /// </summary>
    /// <param name="playerAction"></param>
    /// <returns></returns>
    private Vector3 GetMovePos(PlayerActionData playerAction)
    {
        var pos = new Vector3(playerAction.pos_x, playerAction.pos_y, playerAction.pos_z);
        pos.z += (playerAction.way == "up") ? playerAction.range : 0;
        pos.z -= (playerAction.way == "down") ? playerAction.range : 0;
        pos.x -= (playerAction.way == "left") ? playerAction.range : 0;
        pos.x += (playerAction.way == "right") ? playerAction.range : 0;

        return pos;
    }

    private Vector3 GetFingerMovePos(PlayerActionData playerAction){
        float deltatime = Time.deltaTime * speed;
        Debug.Log("test "+playerAction.pos_y);
        return new Vector3 (playerAction.pos_x * deltatime, 0, playerAction.pos_y * deltatime);
    }
}