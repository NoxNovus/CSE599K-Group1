# CSE 599K — Environment Setup Guide

You are expected to develop and test their code on our remote server equipped with **H100 GPUs**. This guide will walk you through setting up your development environment.

---

## 1. Install VSCode (Recommended)

We recommend using **Visual Studio Code (VSCode)** for development. Other editors (Cursor, Vim, etc.) are also fine, but we do not provide official support for them.

1. Download and install VSCode:  
   👉 [https://code.visualstudio.com/](https://code.visualstudio.com/)

2. Open VSCode and click the **Extensions** icon <img src="extension.png" alt="extension" width="40"/> on the left sidebar.

3. Search for and install the **Remote - SSH** extension.

4. Click the **Remote Connection** icon <img src="connect.png" alt="connect" width="40"/> in the bottom-left corner of the window.

5. Select **Connect to Host** from the dropdown.

6. Click **Add New SSH Host**.

7. Enter the following command:
   ```
   ssh group@ptc.cs.washington.edu -p 599<groupid>
   ```
   For example, if you are in **group 11**, use:
   ```
   ssh group@ptc.cs.washington.edu -p 59911
   ```

8. Choose one of the SSH configuration files to update (the first option usually works).

---

## 2. Install VPN (for off-campus access)

If you're working off-campus, you must connect through the UW VPN:

1. Download and install the UW VPN from:  
   👉 [https://uwconnect.uw.edu/it?id=kb_article_view&sysparm_article=KB0034247](https://uwconnect.uw.edu/it?id=kb_article_view&sysparm_article=KB0034247)

2. Connect to the VPN using your UW NetID credentials.

---

## 3. Connect to the Server

1. In VSCode, click the **Remote Connection** icon again and select **Connect to Host**.

2. You should now see `ptc.cs.washington.edu` as an available option — select it.

3. Choose **Linux** when prompted for the platform.

4. Enter the password (which will be provided to you via email).

---

## 4. Set Up the Software Environment

1. Navigate to `/home/group/code`. We have already cloned the course repository there.

   📌 Regularly run the following command to keep it up to date:
   ```bash
   git pull
   ```

2. We recommend using a Python environment manager (e.g., `uv`, `conda`, `venv`, etc.).

3. The course dependencies are listed in:
   ```
   /home/group/code/CSE599K_SP25_Code/requirements.txt
   ```

4. You can install them using any preferred method. Example with `pip`:
   ```bash
   uv pip install -r requirements.txt
   ```

5. We have pre-installed `uv` and created a virtual environment for you:
   ```
   /home/group/code/CSE599K_SP25_Code/.venv/
   ```

6. To activate the environment, run:
   ```bash
   source /home/group/code/CSE599K_SP25_Code/.venv/bin/activate
   ```

---

Let us know if you encounter any issues or need further support!
