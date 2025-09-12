### 2025/9/10
1. Although today try to use more strict and practicle process to debug, the error of the cli still happening again.(Maybe is the original code quality is not well, spending lots of time on the debugging and AI Coding made the codes more complicated.) Like below:
```
 GPT-5-mini ECTD pipeline completed successfully for Iteration 3!        
📂 Results available in: d:\AboutCoding\AI_Research\GraphJudge_TextToKG_CLI\datasets\KIMI_result_DreamOf_RedChamber\Graph_Iteration3
🔄 You can now run the next iteration or proceed to semantic graph generation.
[SUCCESS] Pipeline execution completed successfully!
[LOG] Terminal progress log saved to: ..\docs\Iteration_Terminal_Progress\gpt5mini_entity_iteration_20250910_215621.txt
🔍 Checking location: d:\AboutCoding\AI_Research\GraphJudge_TextToKG_CLI\datasets\KIMI_result_DreamOf_RedChamber\Graph_Iteration3
   ✗ test_entity.txt (missing, 0 bytes)
   ✗ test_denoised.target (missing, 0 bytes)
🔍 Checking location: d:\AboutCoding\AI_Research\GraphJudge_TextToKG_CLI\chat\datasets\KIMI_result_DreamOf_RedChamber\Graph_Iteration3
   ✗ test_entity.txt (missing, 0 bytes)
   ✗ test_denoised.target (missing, 0 bytes)
🔍 Checking location: d:\AboutCoding\AI_Research\GraphJudge_TextToKG_CLI\docs\Iteration_Report\Iteration3\results\ectd
   ✗ test_entity.txt (missing, 0 bytes)
   ✗ test_denoised.target (missing, 0 bytes)
❌ Output files not found in any checked locations:
   Actual: d:\AboutCoding\AI_Research\GraphJudge_TextToKG_CLI\datasets\KIMI_result_DreamOf_RedChamber\Graph_Iteration3
   Primary: d:\AboutCoding\AI_Research\GraphJudge_TextToKG_CLI\chat\datasets\KIMI_result_DreamOf_RedChamber\Graph_Iteration3
   Legacy: d:\AboutCoding\AI_Research\GraphJudge_TextToKG_CLI\docs\Iteration_Report\Iteration3\results\ectd
   Note: Files must exist AND have non-zero size

 ECTD stage completed but missing expected output files!
ECTD stage failed after 526.6s
❌ ECTD failed: Missing expected output files

❌ Pipeline failed at stage: ectd

============================================================
❌ PIPELINE FAILED
📍 Failed at stage: ectd
🔄 To resume: use --start-from-stage ectd
============================================================
```
So, for the sake of just combine function of the `run_entity.py` and `run_triple.py` and `run_gj.py` to be a pipeline, tentively abandon the cli plan. Change to the streamlit plan, change to use streamlit to implement this function will be more smoothly to accomplish AI Function.

---

### 2025/9/12

- Claude code is a good ai coding agent, better than the cursor and github copilot, but it limit 45-50 times of conversationing to ai, so it's necessary to write the detail spec.md and Task.md during developing and when ask you to commmand , need to choose "no ask, auto run" to save the 5 hour 45-50 times conversation credits. 

- And I spend too much credit in the morning to solve the problem that Claude code editted contents wouldn't save to the file. The solution is here, when the first time to use Claude Code, can following below to prevent the same problem :
1) lower the Claude Code edition and update to the latest edition:
   ```bash
   # Completely delete the Claude Code (if already installed)
   npm uninstall -g @anthropic-ai/claude-code

   # install the `1.0.77` edition
   npm install -g @anthropic-ai/claude-code@1.0.77 --save-exact

   # Then update to the latest
   npm install -g @anthropic-ai/claude-code@lates

   ```
2) Ensure not install the extention `Prettier`.
3) Add the two lines below in the heading of `Claude.md`(This file need to set in the Claude Code initially, and it will in your project root folder.):
   - Verify the file status before editing.
   - If encountering storage issues, please reload the file.
4) The permission setting
   - Open the Claude Code.
   - Enter `/permissions`
   - Choose `Allow`
   - Enter the lines below to add the instruction:
     ```
     Always allow "Edit" tool
     Always allow "CreateFile" tool
     ```
5) Test the result
   In claude code enter : "請幫我加入編輯 test.js 檔案，加入一行註解"。 If succeed, the file will show in the root folder.
6) Appendix : If still failed
   - Use VS Code in the command enter : `code %USERPROFILE%\.claude\settings.json`
   - Adding the below to the json file:
   ```
   {
     "allowedTools": {
       "Edit": "always",
       "CreateFile": "always",
       "ReadFile": "always"
      },
     "autoApprove": {
       "edit": true,
       "bash": ["git commit", "git add", "npm install"]
      }
   }

   ```

