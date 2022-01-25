using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class TitleManager : MonoBehaviour
{
    // ユーザー名の入力テキスト
    public InputField IpfUserName;

    /// <summary>
    /// ログインボタン押下時の処理 
    /// </summary>
    public void OnClickLoginButton()
    {
        // 入力したユーザー名の取得 
        UserLoginData.userName = IpfUserName.text;

        // プレイ画面へ遷移
        SceneManager.LoadScene("PlayScene");
    }

    /// <summary>
    /// アプリ終了ボタン押下時の処理
    /// </summary>
    public void OnClickExitButton()
    {
        Application.Quit();
    }
}